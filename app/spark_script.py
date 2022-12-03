import os
import argparse as ap
from argparse import RawDescriptionHelpFormatter

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, \
    GBTClassifier, MultilayerPerceptronClassifier, LinearSVC, NaiveBayes
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, DecisionTreeRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, isnan, when, count, min, max
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.config("spark.driver.memory", "12g").getOrCreate()


class DataLoader:

    @staticmethod
    def load_years(years: list):
        df = DataLoader._load_all_years(years)
        return df

    @staticmethod
    def _load_all_years(years: list):
        df = DataLoader._load_one_year(years[0])
        if len(years) == 1:
            return df

        for y in years[1:]:
            df = df.unionByName(DataLoader._load_one_year(y))
        return df

    @staticmethod
    def _load_one_year(year: int):
        return spark.read.csv(f'app/data/{year}.csv', header=True)


class DataCleaner:
    ForbiddenVariables = ['ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted', 'CarrierDelay',
                          'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']

    @staticmethod
    def clean(df):
        df = DataCleaner._remove_forbidden(df)
        df = DataCleaner._remove_cancelled(df)
        df = DataCleaner._remove_nulls(df)
        return df

    @staticmethod
    def _remove_forbidden(df):
        return df.drop(*DataCleaner.ForbiddenVariables)

    @staticmethod
    def _remove_cancelled(df):
        df = df.filter(df.Cancelled == "0")
        df = df.drop('Cancelled', 'CancellationCode')
        return df

    @staticmethod
    def _remove_nulls(df):
        # df = df.filter(col('ArrDelay').isNotNull())
        # df = df.filter(col('CRSElapsedTime').isNotNull())
        df = df.dropna()
        return df


class DataTransformer:
    IntColumns = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'CRSElapsedTime', 'ArrDelay', 'DepDelay', 'Distance', 'TaxiOut']

    @staticmethod
    def transform(df, class_interv):
        df = DataTransformer._do_transform_time_to_mins(df)
        df = DataTransformer._do_cast_ints(df)
        df = DataTransformer._one_hot_encode(df)
        df = DataTransformer._prepare_features_cols(df)
        df = DataTransformer._keep_train_cols_only(df)
        df = DataTransformer._prepare_label_categories(df, class_interv)
        return df

    @staticmethod
    def _do_transform_time_to_mins(df):
        def time_to_mins(t: str):
            t = t.zfill(4)
            return int(t[:2]) * 60 + int(t[2:])

        time_to_mins_udf = F.udf(time_to_mins, IntegerType())
        df = df.withColumn('DepTime', time_to_mins_udf('DepTime'))
        df = df.withColumn('CRSDepTime', time_to_mins_udf('CRSDepTime'))
        df = df.withColumn('CRSArrTime', time_to_mins_udf('CRSArrTime'))
        return df

    @staticmethod
    def _do_cast_ints(df):
        for c in DataTransformer.IntColumns:
            df = df.withColumn(c, col(c).cast('int'))
        df = df.dropna()  # Casting may introduce some new null values in
        return df

    @staticmethod
    def _one_hot_encode(df):
        categorical_columns = [item[0] for item in df.dtypes if item[1].startswith('string')]
        for c in categorical_columns:
            string_indexer = StringIndexer(inputCol=c, outputCol=c + "Index")
            df = string_indexer.fit(df).transform(df)

        for column in categorical_columns:
            one_hot_encoder = OneHotEncoder(inputCol=column + "Index", outputCol=column + "_vec")
            df = one_hot_encoder.fit(df).transform(df)
        return df

    @staticmethod
    def _prepare_features_cols(df):
        input_columns = ['DepTime', 'CRSDepTime', 'CRSArrTime', 'CRSElapsedTime', 'DepDelay', 'Distance', 'TaxiOut']
        output_column = "features"

        vector_a = VectorAssembler(inputCols=input_columns, outputCol=output_column)
        df = vector_a.transform(df)
        return df

    @staticmethod
    def _keep_train_cols_only(df):
        return df.select(['features', 'ArrDelay']).withColumnRenamed('ArrDelay', 'regressionLabel')

    @staticmethod
    def _prepare_label_categories(df, interval):
        def category_from_delay(delay: int, interval: int):
            if delay <= 0:
                return 0
            if delay >= 180:
                return 180 // interval + 1
            return delay // interval + 1

        category_from_delay_udf = F.udf(lambda x: category_from_delay(x, interval), IntegerType())
        df = df.withColumn('classLabel', category_from_delay_udf('regressionLabel'))
        df.groupBy('classLabel').count().show()
        return df


class Trainer:
    available_models = {}
    cross_validators = {}
    evaluator = None
    parallelism = 8

    @classmethod
    def train(cls, df, selected_models, use_cv):
        if use_cv:
            models = {k: cls.cross_validators[k].fit(df) for k in selected_models}
        else:
            trainers = {k: cls.available_models[k] for k in selected_models}
            models = {k: trainers[k].fit(df) for k in trainers}
        return models

    @classmethod
    def test(cls, df, models) -> tuple:
        predictions = {k: models[k].transform(df) for k in models}
        evaluations = {k: cls.evaluator.evaluate(predictions[k]) for k in predictions}
        return predictions, evaluations


class RegressionTrainer(Trainer):
    available_models = {
        'lr': LinearRegression(labelCol='regressionLabel'),
        'dtr': DecisionTreeRegressor(labelCol='regressionLabel'),
        'rfr': RandomForestRegressor(labelCol='regressionLabel'),
        'gbtr': GBTRegressor(labelCol='regressionLabel')
    }
    grids = {
        'lr': ParamGridBuilder().addGrid(available_models['lr'].regParam, [0.0, 0.1]).build(),
        'dtr': ParamGridBuilder().addGrid(available_models['dtr'].maxDepth, [3, 5, 7]).addGrid(available_models['dtr'].maxBins, [8, 16, 32]).build(),
        'rfr': ParamGridBuilder().addGrid(available_models['dtr'].maxDepth, [3, 5, 7]).addGrid(available_models['dtr'].maxBins, [8, 16, 32]).build(),
        'gbtr': ParamGridBuilder().addGrid(available_models['dtr'].maxDepth, [3, 5, 7]).addGrid(available_models['dtr'].maxBins, [8, 16, 32]).build(),
    }
    evaluator = RegressionEvaluator(labelCol='regressionLabel')
    cross_validators = {
        'lr': CrossValidator(estimator=available_models['lr'], estimatorParamMaps=grids['lr'], evaluator=evaluator, parallelism=Trainer.parallelism),
        'dtr': CrossValidator(estimator=available_models['dtr'], estimatorParamMaps=grids['dtr'], evaluator=evaluator, parallelism=Trainer.parallelism),
        'rfr': CrossValidator(estimator=available_models['rfr'], estimatorParamMaps=grids['rfr'], evaluator=evaluator, parallelism=Trainer.parallelism),
        'gbtr': CrossValidator(estimator=available_models['gbtr'], estimatorParamMaps=grids['gbtr'], evaluator=evaluator, parallelism=Trainer.parallelism),
    }


class ClassificationTrainer(Trainer):
    available_models = {
        'mlr': LogisticRegression(labelCol='classLabel'),
        'dtc': DecisionTreeClassifier(labelCol='classLabel'),
        'rfc': RandomForestClassifier(labelCol='classLabel'),
        'gbtc': GBTClassifier(labelCol='classLabel'),
        'mlpc': MultilayerPerceptronClassifier(labelCol='classLabel'),
        'lsvc': LinearSVC(labelCol='classLabel'),
        'nbc': NaiveBayes(labelCol='classLabel')
    }
    grids = {
        'mlr': ParamGridBuilder().addGrid(available_models['mlr'].regParam, [0.0, 0.1]).build(),
        'dtc': ParamGridBuilder().addGrid(available_models['dtc'].maxDepth, [3, 5, 7]).addGrid(available_models['dtc'].maxBins, [8, 16, 32]).build(),
        'rfc': ParamGridBuilder().addGrid(available_models['rfc'].maxDepth, [3, 5, 7]).addGrid(available_models['rfc'].maxBins, [8, 16, 32]).build(),
        'gbtc': ParamGridBuilder().addGrid(available_models['gbtc'].maxDepth, [3, 5, 7]).addGrid(available_models['gbtc'].maxBins, [8, 16, 32]).build(),
        'mlpc': ParamGridBuilder().addGrid(available_models['mlpc'].tol, [1e-06, 1e-07]).build(),
        'lsvc': ParamGridBuilder().addGrid(available_models['lsvc'].regParam, [0.0, 0.1]).addGrid(available_models['lsvc'].tol, [1e-06, 1e-07]).build(),
        'nbc': ParamGridBuilder().addGrid(available_models['nbc'].smoothing, [0.8, 1.0]).build(),
    }
    evaluator = MulticlassClassificationEvaluator()
    cross_validators = {
        'mlr': CrossValidator(estimator=available_models['mlr'], estimatorParamMaps=grids['mlr'], evaluator=evaluator, parallelism=Trainer.parallelism),
        'dtc': CrossValidator(estimator=available_models['dtc'], estimatorParamMaps=grids['dtc'], evaluator=evaluator, parallelism=Trainer.parallelism),
        'rfc': CrossValidator(estimator=available_models['rfc'], estimatorParamMaps=grids['rfc'], evaluator=evaluator, parallelism=Trainer.parallelism),
        'gbtc': CrossValidator(estimator=available_models['gbtc'], estimatorParamMaps=grids['gbtc'], evaluator=evaluator, parallelism=Trainer.parallelism),
        'mlpc': CrossValidator(estimator=available_models['mlpc'], estimatorParamMaps=grids['mlpc'], evaluator=evaluator, parallelism=Trainer.parallelism),
        'lsvc': CrossValidator(estimator=available_models['lsvc'], estimatorParamMaps=grids['lsvc'], evaluator=evaluator, parallelism=Trainer.parallelism),
        'nbc': CrossValidator(estimator=available_models['nbc'], estimatorParamMaps=grids['nbc'], evaluator=evaluator, parallelism=Trainer.parallelism),
    }


def run_spark(years: list = [], reg_models: list = [], class_models: list = [], class_interv: int = 10, use_cross_val=True):

    df = DataLoader.load_years(years)
    df = DataCleaner.clean(df)
    df = DataTransformer.transform(df, class_interv)
    df_train, df_test = df.randomSplit([0.7, 0.3])

    reg_trainer = RegressionTrainer()
    clas_trainer = ClassificationTrainer()
    reg_trained = reg_trainer.train(df_train, reg_models, use_cross_val)
    class_trained = clas_trainer.train(df_train, class_models, use_cross_val)

    reg_preds, reg_evals = reg_trainer.test(df_test, reg_trained)
    cls_preds, cls_evals = clas_trainer.test(df_test, class_trained)

    df.printSchema()
    print(df.count())
    df.show(20)
    print(reg_evals)
    print(cls_evals)

    for rp in reg_preds.values():
        rp.show()
    for cp in cls_preds.values():
        cp.show()

    for k in reg_evals:
        print(k, reg_evals[k])

    exit(0)


if __name__ == '__main__':
    SCRIPT_NAME = os.path.basename(__file__)
    # TODO Add the other models
    parser = ap.ArgumentParser(f'spark-submit {SCRIPT_NAME}', formatter_class=RawDescriptionHelpFormatter,
                               description='Spark Application to predict arrival delays in plane flights.\
                                    \nList of all available regression methods:\
                                    \n    - lr: Linear Regression\
                                    \nList of all available classification methods:\
                                    \n    - lr: Linear Regression\
        ')
    parser.add_argument('-y', '--years', type=str,
                        help='Comma-separated list of years data to use. Values in range [1987, 2008]. Values between '
                             'commas can be simple or hyphen separated for ranges. e.g. 1991-1994,1999,2001-2003')
    parser.add_argument('-r', '--regressions', type=str, default='all',
                        help='Comma-separated list of regression methods to use. See help for all methods (default: %(default)s)')
    parser.add_argument('-c', '--classifications', type=str, default='all',
                        help='Comma-separated list of classification methods to use (default: %(default)s)')
    parser.add_argument('-ci', '--classification-interval', type=int, default=10,
                        help='When using classification methods, the interval of the categories in minutes (default: %(default)s)')
    parser.add_argument('-cv', '--cross-validation', default=False, action='store_true',
                        help='Enable cross validation in all trained models')
    args = parser.parse_args()


    def years_parser(years_str: str) -> list:
        years_list = []
        for y in years_str.split(','):
            if '-' not in y:
                years_list.append(int(y))
            else:
                i, e = tuple(y.split('-'))
                years_list += [n for n in range(int(i), int(e) + 1)]
        return years_list

    def models_parser(inp_str: str, models) -> list:
        if inp_str == 'none':
            return []
        if inp_str == 'all':
            return [k for k in models]
        return inp_str.split(',')


    years = years_parser(args.years.strip())
    regressors = models_parser(args.regressions.strip(), RegressionTrainer.available_models)
    classifiers = models_parser(args.classifications.strip(), ClassificationTrainer.available_models)
    class_int = args.classification_interval
    run_spark(years=years, reg_models=regressors, class_models=classifiers, class_interv=class_int)
