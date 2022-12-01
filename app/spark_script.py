import os
import argparse as ap
import sys
from argparse import RawDescriptionHelpFormatter

from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, \
    GBTClassifier, MultilayerPerceptronClassifier, LinearSVC, NaiveBayes
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, DecisionTreeRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, isnan, when, count
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.getOrCreate()


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
    def transform(df):
        df = DataTransformer._do_transform_time_to_mins(df)
        df = DataTransformer._do_cast_ints(df)
        df = DataTransformer._one_hot_encode(df)
        df = DataTransformer._prepare_features_cols(df)
        df = DataTransformer._keep_train_cols_only(df)
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
        df = df.withColumnRenamed('Month', 'MonthIndex')
        df = df.withColumnRenamed('DayOfWeek', 'DayOfWeekIndex')

        df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()

        for column in categorical_columns + ['Month', 'DayOfWeek']:
            one_hot_encoder = OneHotEncoder(inputCol=column + "Index", outputCol=column + "_vec")
            df = one_hot_encoder.fit(df).transform(df)
        return df

    @staticmethod
    def _prepare_features_cols(df):
        input_columns = ['Year', 'Month_vec', 'DayofMonth', 'DayOfWeek_vec', 'DepTime', 'CRSDepTime', 'CRSArrTime',
                         'CRSElapsedTime', 'DepDelay', 'Distance', 'TaxiOut', 'UniqueCarrier_vec', 'FlightNum_vec',
                         'TailNum_vec', 'Origin_vec', 'Dest_vec']
        output_column = "features"

        vector_a = VectorAssembler(inputCols=input_columns, outputCol=output_column)
        df = vector_a.transform(df)
        return df

    @staticmethod
    def _keep_train_cols_only(df):
        return df.select(['features', 'ArrDelay']).withColumnRenamed('ArrDelay', 'label')


class Trainer:
    available_models = {}
    evaluator = None

    @classmethod
    def train(cls, df, selected_models):
        trainers = {k: cls.available_models[k] for k in selected_models}
        models = {k: trainers[k].fit(df) for k in trainers}
        return models

    @classmethod
    def test(cls, df, models):
        predictions = {k: models[k].transform(df) for k in models}
        evaluations = {k: cls.evaluator.evaluate(predictions[k]) for k in predictions}
        return evaluations


class RegressionTrainer(Trainer):
    available_models = {
        'lr': LinearRegression(featuresCol='features', labelCol='label'),
        'dtr': DecisionTreeRegressor(),
        'rfr': RandomForestRegressor(featuresCol='features'),
        'gbtr': GBTRegressor()
    }
    evaluator = RegressionEvaluator()


class ClassificationTrainer(Trainer):
    available_models = {
        'mlr': LogisticRegression(),
        'dtc': DecisionTreeClassifier(),
        'rfc': RandomForestClassifier(),
        'gbtc': GBTClassifier(),
        'mlpc': MultilayerPerceptronClassifier(),
        'lsvc': LinearSVC(),
        'nbc': NaiveBayes()
    }
    evaluator = MulticlassClassificationEvaluator()


'''
class RegressionHelper:
    regressions = {
        'lr': LinearRegression(featuresCol='features', labelCol='label'),
        'dtr': DecisionTreeRegressor(),
        'rfr': RandomForestRegressor(featuresCol='features'),
        'gbtr': GBTRegressor()
    }
    evaluator = RegressionEvaluator()

    @staticmethod
    def train(df, selected_models) -> dict:
        regressors = {k: RegressionHelper.regressions[k] for k in selected_models}
        models = {k: regressors[k].fit(df) for k in regressors}
        return models

    @staticmethod
    def test(df, models) -> dict:
        predictions = {k: models[k].transform(df) for k in models}
        evaluations = {k: RegressionHelper.evaluator.evaluate(df, predictions[k]) for k in predictions}
        return evaluations


class ClassificationHelper:
    classifications = {
        'mlr': LogisticRegression(),
        'dtc': DecisionTreeClassifier(),
        'rfc': RandomForestClassifier(),
        'gbtc': GBTClassifier(),
        'mlpc': MultilayerPerceptronClassifier(),
        'lsvc': LinearSVC(),
        'nbc': NaiveBayes()
    }
    evaluator = MulticlassClassificationEvaluator()

    @staticmethod
    def train(df, selected_models) -> dict:
        classifiers = {k: ClassificationHelper.classifications[k] for k in selected_models}
        models = {k: classifiers[k].fit(df) for k in classifiers}
        return models

    @staticmethod
    def test(df, models) -> dict:
        predictions = {k: models[k].transform(df) for k in models}
        evaluations = {k: ClassificationHelper.evaluator.evaluate(df, predictions[k]) for k in predictions}
        return evaluations
'''


def run_spark(years: list = [], reg_models: list = [], class_models: list = [], class_interv: list = []):
    print(years)

    df = DataLoader.load_years(years)
    df = DataCleaner.clean(df)
    df = DataTransformer.transform(df)
    df_train, df_test = df.randomSplit([0.7, 0.3])

    reg_trainer = RegressionTrainer()
    clas_trainer = ClassificationTrainer()
    reg_trained = reg_trainer.train(df_train, reg_models)
    #class_trained = clas_trainer.train(df_train, class_models)

    # TODO Classification models

    regression_evaluations = reg_trainer.test(df_test, reg_trained)
    #classification_evaluations = clas_trainer.test(df_test, class_trained)

    df.printSchema()
    print(df.count())
    df.show(20)
    print(regression_evaluations)


if __name__ == '__main__':
    SCRIPT_NAME = os.path.basename(__file__)
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
        if inp_str == 'all':
            return [k for k in models]
        return inp_str.split(',')


    years = years_parser(args.years.strip())
    regressors = models_parser(args.regressions.strip(), RegressionTrainer.available_models)
    classifiers = models_parser(args.classifications.strip(), ClassificationTrainer.available_models)
    class_int = args.classification_interval
    run_spark(years=years, reg_models=regressors, class_models=classifiers, class_interv=class_int)
