import numpy as np
import pandas as pd

from train_models import *

from pyspark import SparkContext
from pyspark.sql import SQLContext
from sparkdl import readImages
# import sparkdl
from pyspark.sql.functions import lit

from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from pyspark.ml.feature import StringIndexer

sc = SparkContext.getOrCreate()
sqlCtx = SQLContext(sc)
img_dir = "/Users/macbook/Desktop/presidents_200/data"

obama_df = readImages(img_dir + "/obama").withColumn("label", lit(1))
trump_df = readImages(img_dir + "/trump").withColumn("label", lit(0))
obama_train, obama_test = obama_df.randomSplit([0.6, 0.4], seed = 314)
trump_train, trump_test = trump_df.randomSplit([0.6, 0.4], seed = 314)

print ("IMAGES IMPORTED")

#dataframe for training a classification model
train_df = obama_train.unionAll(trump_train)

#dataframe for testing the classification model
test_df = obama_test.unionAll(trump_test)
print ("CREATED TEST TRAIN DATASETS")

def main():
    randomforestPipeline(train_df, test_df)
    logregPipeline(train_df, test_df)
    vgg16Pipe(train_df, test_df)
    vgg19Pipe(train_df, test_df)

if __name__ == '__main__':
    main()
