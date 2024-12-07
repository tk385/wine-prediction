import sys
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler, PolynomialExpansion
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql import DataFrame
import boto3
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_data(df: DataFrame) -> DataFrame:
    """Clean and prepare DataFrame by casting all columns to double."""
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))

def save_single_file(dataframe: DataFrame, output_path: str):
    """Save the DataFrame as a single CSV file in S3."""
    temp_path = "s3://wineprec/datasets/temp_test_data"
    dataframe.coalesce(1).write.mode("overwrite").csv(temp_path, header=True, sep=";")

    s3 = boto3.resource('s3')
    bucket_name = "wineprec"
    temp_folder = "datasets/temp_test_data"
    target_path = "datasets/TestDataset.csv"

    bucket = s3.Bucket(bucket_name)
    part_file = next((obj.key for obj in bucket.objects.filter(Prefix=temp_folder) if obj.key.endswith(".csv")), None)

    if part_file:
        bucket.copy({'Bucket': bucket_name, 'Key': part_file}, target_path)
        bucket.objects.filter(Prefix=temp_folder).delete()

if __name__ == "__main__":
    try:
        logger.info("Starting Spark session...")
        spark = SparkSession.builder \
            .appName('Wine_Quality_Prediction_RF') \
            .config("spark.sql.shuffle.partitions", "50") \
            .getOrCreate()

        logger.info("Setting Spark log level...")
        spark.sparkContext.setLogLevel('ERROR')

        input_path = sys.argv[1] if len(sys.argv) > 1 else "s3://wineprec/datasets/TrainingDataset.csv"
        valid_path = sys.argv[2] if len(sys.argv) > 2 else "s3://wineprec/datasets/ValidationDataset.csv"
        model_output_path = sys.argv[3] if len(sys.argv) > 3 else "s3://winepredc/models/winemodel"

        logger.info("Loading and cleaning training dataset...")
        train_df = spark.read.format("csv").option('header', 'true').option("sep", ";").load(input_path)
        cleaned_train_df = clean_data(train_df)

        logger.info("Splitting the dataset into training and testing...")
        train_data, test_data = cleaned_train_df.randomSplit([0.9, 0.1], seed=42)

        logger.info("Saving test dataset to S3 as a single file...")
        save_single_file(test_data, "s3://wineprec/datasets/TestDataset.csv")

        logger.info("Loading and cleaning validation dataset...")
        valid_df = spark.read.format("csv").option('header', 'true').option("sep", ";").load(valid_path)
        valid_data = clean_data(valid_df)

        logger.info("Building pipeline and cross-validation setup...")
        all_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                        'pH', 'sulphates', 'alcohol']

        assembler = VectorAssembler(inputCols=all_features, outputCol='features')
        scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
        poly_expansion = PolynomialExpansion(inputCol="scaledFeatures", outputCol="polyFeatures", degree=2)
        indexer = StringIndexer(inputCol="quality", outputCol="label")

        rf = RandomForestClassifier(labelCol='label', featuresCol='polyFeatures', seed=42)
        pipeline = Pipeline(stages=[assembler, scaler, poly_expansion, indexer, rf])

        paramGrid = ParamGridBuilder() \
            .addGrid(rf.numTrees, [100, 200]) \ 
            .addGrid(rf.maxDepth, [5, 10, 20]) \
            .addGrid(rf.minInstancesPerNode, [1, 5]) \
            .addGrid(rf.featureSubsetStrategy, ['auto', 'sqrt']) \
            .build()

        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

        crossval = CrossValidator(estimator=pipeline, 
                                  estimatorParamMaps=paramGrid, 
                                  evaluator=evaluator, 
                                  numFolds=2,
                                  parallelism=4)

        logger.info("Training with Cross-Validation...")
        cv_model = crossval.fit(train_data)

        logger.info("Evaluating the model on validation data...")
        predictions = cv_model.bestModel.transform(valid_data)
        accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
        f1_score = evaluator.evaluate(predictions)

        logger.info(f"Best Model Test Accuracy: {accuracy}")
        logger.info(f"Best Model Weighted F1 Score: {f1_score}")

        logger.info(f"Saving the best model to {model_output_path}...")
        cv_model.bestModel.write().overwrite().save(model_output_path)

        spark.stop()

    except Exception as e:
        logger.error(f"An error occurred: {e}")
