import warnings
warnings.filterwarnings('ignore')
from pyspark.sql import SQLContext
from pyspark import SparkContext
from pyspark.sql.functions import *
import pyspark.sql.functions as f
from pyspark.sql.types import TimestampType
from datetime import datetime
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
import pandas as pd


def model_trainer(final_df, train_year, train_start_month, train_end_month, test_year, test_month):
    """
    :param raw_df: original dataframe
    :param train_year: the year of the data used for training
    :param train_start_month: the first month of the data used for training
    :param train_end_month: the last month of the data used for training
    :param test_year: the year of the data used for training
    :param test_month: month of the data used for testing
    :return mape: the MAPE in the test set
    """
    # select features
    target = 'profit'
    features = ['bar_num', 'var12', 'var13', 'var14', 'var15', 'var16', 'var17', 'var18', 'var23', 'var24', 'var25',
                'var26', 'var27', 'var28', 'var34', 'var35', 'var36', 'var37', 'var38', 'var45', 'var46', 'var47',
                'var48', 'var56', 'var57', 'var58', 'var67', 'var68', 'var78',
                'lag0_profit', 'lag1_profit', 'lag2_profit', 'lag3_profit', 'lag4_profit', 'lag5_profit',
                'mean_profit', 'std_profit'
                ]
    # make model pipeline
    assembler = VectorAssembler(inputCols=features, outputCol='features').setHandleInvalid('skip')
    rf = RandomForestRegressor(labelCol="profit", featuresCol="features")
    pipeline = Pipeline(stages=[assembler, rf])
    # train & test split
    train = (final_df
             .filter(f.col('Year') == train_year)
             .filter((f.col('Month') >= train_start_month) & (f.col('Month') <= train_end_month))
             .select('profit', *(c for c in features)))
    test = (final_df
            .filter(f.col('Year') == test_year)
            .filter(f.col('Month') == test_month)
            .select('profit', *(c for c in features)))
    # fit the model pipeline
    model = pipeline.fit(train)
    predictions = model.transform(test)
    predictions2 = predictions.select(f.col("profit").cast("Float"), f.col("prediction"))
    predictions2 = predictions2.withColumn('ape',
                                           100 * (abs((f.col('profit') - f.col('prediction')) / f.col('profit')))
                                           )
    mape_df = predictions2.select(
        f.mean(f.col('ape')).alias('mape')
    ).collect()
    mape = mape_df[0]['mape']
    return mape


if __name__ == "__main__":
    sc = SparkContext()
    sqlcontext = SQLContext(sc)
    path = 's3://msia431hw3/full_data.csv'
    mydata = sqlcontext.read.csv(path, header=True)
    # change column types from string to integer
    df = mydata.select('time_stamp', *(col(c).cast(IntegerType()).alias(c) for c in mydata.columns[1:]))
    # extract month and year from time_stamp_column
    timestamp_format = 'yyyy-MM-dd'
    df = (df
          .withColumn('DateTime', unix_timestamp(col('time_stamp'), timestamp_format).cast('timestamp'))
          .withColumn('Year', year(col('DateTime')))
          .withColumn('Month', month(col('DateTime'))))

    ###################
    # generate features
    ###################
    # Step1: create bar group
    #        group 0: 1<=bar<=10;
    #        group 1: 11<=bar<=20;
    #        group 2: 21<=bar<=30, etc
    # Step2: create features within each bar group
    #        the last profit (lag0), lag1 profit, lag2 profit, lag3 profit, lag4 profit, lag5 profit,
    #        average profit, and the standard deviation of profits
    df = df.orderBy(["trade_id", "bar_num"], ascending=[True, True])
    df = df.withColumn('barGroup', ((f.col('bar_num') - 1) / 10).cast('integer'))
    w = Window().partitionBy('trade_id', 'barGroup').orderBy('barGroup')
    df2 = (df
           .select('trade_id', 'barGroup', 'bar_num', 'profit',
                   f.last("profit", True).over(w).alias('lag0_profit'),
                   f.lag("profit", 1).over(w).alias('lag1_profit'),
                   f.lag("profit", 2).over(w).alias('lag2_profit'),
                   f.lag("profit", 3).over(w).alias('lag3_profit'),
                   f.lag("profit", 4).over(w).alias('lag4_profit'),
                   f.lag("profit", 5).over(w).alias('lag5_profit'),
                   f.mean(col('profit')).over(w).alias('mean_profit'),
                   f.stddev(col('profit')).over(w).alias('std_profit'))
           .filter(col('bar_num') % 10 == 0)
           .orderBy(["trade_id", "barGroup"], ascending=[True, True]))
    # Step 3: add a column for joining features back to the original dataframe
    #         bargroup 1 (bar_number 11-20) will use features generated using profits in bargroup 0 (1-10)
    #         bargroup 2 (bar_number 21-30) will use features generated using profits in bargourp 1 (11-20)
    #         etc.
    df2 = (df2
           .withColumn('next_barGroup', f.col('barGroup') + 1)
           .select('trade_id', 'lag0_profit', 'lag1_profit', 'lag2_profit', 'lag3_profit', 'lag4_profit', 'lag5_profit',
                   'mean_profit', 'std_profit', 'next_barGroup')
           .withColumnRenamed('trade_id', 't_trade_id'))
    # Step 4: join features back to the original data frame
    merged_df = df.join(df2, on=(df.barGroup == df2.next_barGroup) & (df.trade_id == df2.t_trade_id), how='left')
    # Step 5: keep only data for with bar_num >10; there is nothing to predict for bar 1-10
    final_df = merged_df.filter(f.col('bar_num') > 10)

    ###################
    # train models
    ###################
    # train and evaluate the model in a move-forward way
    result = []
    for year in range(2008, 2015):
        result.append(model_trainer(final_df, year, 1, 6, year, 7))
        result.append(model_trainer(final_df, year, 7, 12, year + 1, 1))
    result.append(model_trainer(final_df, 2015, 1, 6, 2015, 7))

    # format model results
    d = {'train_year': [y for y in range(2008, 2015) for i in range(2)] + [2015],
         'train_month': ['1-6', '7-12'] * 7 + ['1-6'],
         'test_year': [2008] + [y for y in range(2009, 2016) for i in range(2)],
         'test_month': [7, 1] * 7 + [7],
         'MAPE': result
         }
    output = pd.DataFrame(data=d)

    # calculate min/max/avg MAP
    d2 = {'AVG_MAPE': [output.MAPE.mean()],
          'MAX_MAPE': [output.MAPE.max()],
          'MIN_MAPE': [output.MAPE.min()]
          }
    output2 = pd.DataFrame(data=d2)

    # save output file to S3
    output.to_csv('s3://msia431hw3/Exercise3.txt', header=True, index=False, sep=',')
    output2.to_csv('s3://msia431hw3/Exercise3.txt', header=True, index=False, sep=',', mode='a')

    sc.stop()
