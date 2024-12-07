import sys
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_data(df):
    """Clean and prepare DataFrame by casting all columns to double."""
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))

if __name__ == "__main__":
    try:
        logger.info("Starting Spark session...")
        spark = SparkSession.builder \
            .appName('Wine_Quality_Test_Model') \
            .config("spark.sql.shuffle.partitions", "50") \
            .getOrCreate()

        logger.info("Setting Spark log level...")
        spark.sparkContext.setLogLevel('ERROR')

        test_data_path = sys.argv[1] if len(sys.argv) > 1 else "s3://wineprec/datasets/TestDataset.csv"
        model_path = sys.argv[2] if len(sys.argv) > 2 else "s3://winepredc/models/winemodel"

        logger.info("Loading and cleaning test dataset...")
        test_df = spark.read.format("csv").option('header', 'true').option("sep", ";").load(test_data_path)
        cleaned_test_df = clean_data(test_df)

        logger.info(f"Loading the trained model from {model_path}...")
        from pyspark.ml.classification import RandomForestClassificationModel
        loaded_model = RandomForestClassificationModel.load(model_path)

        logger.info("Making predictions on the test dataset...")
        predictions = loaded_model.transform(cleaned_test_df)

        logger.info("Evaluating the predictions...")
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
        f1_score = f1_evaluator.evaluate(predictions)

        logger.info(f"Test Dataset Accuracy: {accuracy}")
        logger.info(f"Test Dataset F1 Score: {f1_score}")

        spark.stop()

    except Exception as e:
        logger.error(f"An error occurred during testing: {e}")



