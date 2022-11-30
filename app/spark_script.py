import os
import argparse as ap
from argparse import RawDescriptionHelpFormatter

import pyspark
from pyspark import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import transform, col, isnan, when, count
from pyspark.sql.types import StructType, StructField, FloatType, BooleanType, DoubleType, IntegerType, StringType


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
        df = df.filter(col('ArrDelay').isNotNull())
        df = df.filter(col('CRSElapsedTime').isNotNull())
        return df


class DataTransformer:

    @staticmethod
    def transform(df):
        df = DataTransformer._do_transform_time_to_mins(df)
        # TODO OneHotEncoder
        #df = DataTransformer._prepare_features_cols(df)
        #df = DataTransformer._keep_train_cols_only(df)
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
    def _prepare_features_cols(df):
        input_columns = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime', 'CRSArrTime',
                         'CRSElapsedTime', 'DepDelay', 'Distance', 'TaxiOut', 'UniqueCarrier_vec', 'FlightNum_vec',
                         'TailNum_vec', 'Origin_vec', 'Dest_vec']
        output_column = "features"

        vector_a = VectorAssembler(inputCols=input_columns, outputCol=output_column)
        df = vector_a.transform(df)
        return df

    @staticmethod
    def _keep_train_cols_only(df):
        return df.select(['features', 'ArrDelay']).withColumnRenamed('ArrDelay', 'label')


class RegressionHelper:

    @staticmethod
    def train(df, selected_models) -> list:
        pass

    @staticmethod
    def test(df, models) -> list:
        pass


class ClassificationHelper:

    @staticmethod
    def train(df, selected_models) -> list:
        pass

    @staticmethod
    def test(df, models) -> list:
        pass


def run_spark(years: list = [], reg_models: list = [], class_models: list = [], class_interv: list = []):

    print(years)

    df = DataLoader.load_years(years)
    df = DataCleaner.clean(df)
    df = DataTransformer.transform(df)
    df_train, df_test = df.randomSplit([0.7, 0.3])

    reg_trained = RegressionHelper.train(df_train, reg_models)
    class_trained = ClassificationHelper.train(df_train, reg_models)

    reg_eval = RegressionHelper.test(df_test, reg_trained)
    class_eval = ClassificationHelper.test(df_test, class_trained)

    df.printSchema()
    print(df.count())
    df.show(20)


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
    parser.add_argument('-r', '--regression', type=str, default='all',
                        help='Comma-separated list of regression methods to use. See help for all methods (default: %(default)s)')
    parser.add_argument('-c', '--classification', type=str, default='all',
                        help='Comma-separated list of classification methods to use (default: %(default)s)')
    parser.add_argument('-ci', '--classification-interval', type=int, default=10,
                        help='When using classification methods, the interval of the categories in minutes (default: %(default)s)')
    args = parser.parse_args()

    def years_parser(years_str: str):
        years_list = []
        for y in years_str.split(','):
            if '-' not in y:
                years_list.append(int(y))
            else:
                i, e = tuple(y.split('-'))
                years_list += [n for n in range(int(i), int(e)+1)]
        return years_list

    years = years_parser(args.years.strip())
    run_spark(years)
