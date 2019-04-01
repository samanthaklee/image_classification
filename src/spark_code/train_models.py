from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from pyspark.ml.feature import StringIndexer

def randomforestPipeline(train, test):
    featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
    # labeler = StringIndexer(inputCol='label', outputCol='label')
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")
    p = Pipeline(stages=[featurizer, rf])

    print ("BUILT PIPE")
    p_model = p.fit(train) ## this line causing errors or someting
    print ("MODEL TRAINED")

    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    df = p_model.transform(test)
    #df.show()

    predictionAndLabels = df.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


#also consider the Inception V3 component
def logregPipeline(train, test):

    featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
    # labeler = StringIndexer(inputCol='label', outputCol='label')
    lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
    p = Pipeline(stages=[featurizer, lr])

    print ("BUILT PIPE")
    p_model = p.fit(train) ## this line causing errors or someting
    print ("MODEL TRAINED")

    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    df = p_model.transform(test)
    #df.show()

    predictionAndLabels = df.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


#pre trained neuralnet comparison
def vgg16Pipe(train, test):

    featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="VGG16")
    # labeler = StringIndexer(inputCol='label', outputCol='label')
    lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
    p = Pipeline(stages=[featurizer, lr])

    print ("BUILT PIPE")
    p_model = p.fit(train) ## this line causing errors or someting
    print ("MODEL TRAINED")

    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    df = p_model.transform(test)
    #df.show()

    predictionAndLabels = df.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))


def vgg19Pipe(train, test):

    featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="VGG19")
    # labeler = StringIndexer(inputCol='label', outputCol='label')
    lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
    p = Pipeline(stages=[featurizer, lr])

    print ("BUILT PIPE")
    p_model = p.fit(train) ## this line causing errors or someting
    print ("MODEL TRAINED")

    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    df = p_model.transform(test)
    #df.show()

    predictionAndLabels = df.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
