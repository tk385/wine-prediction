import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col

def clean_data(df):
    """Clean and cast data to double."""
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))

if __name__ == "__main__":
    # Initialize Spark session with S3 access configurations
    spark = SparkSession.builder \
        .appName('Wine_Quality_Test_Prediction') \
        .getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

    # Read arguments
    if len(sys.argv) < 3:
        print("Usage: quilitytestdataprediction.py <test_dataset_path> <model_path>")
        sys.exit(-1)

    test_data_path = sys.argv[1]  # Path to TestDataset.csv in S3
    model_path = sys.argv[2]      # Path to the saved trained model in S3

    print(f"Loading test data from: {test_data_path}")
    print(f"Loading model from: {model_path}")

    # Load test dataset
    try:
        test_df = (spark.read
                      .format("csv")
                      .option('header', 'true')
                      .option("sep", ";")
                      .load(test_data_path))
    except Exception as e:
        print(f"Error loading test dataset from {test_data_path}: {e}")
        spark.stop()
        sys.exit(1)

    # Clean the test data
    test_data = clean_data(test_df)

    # Load the trained model
    try:
        trained_model = PipelineModel.load(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        spark.stop()
        sys.exit(1)

    # Generate predictions
    print("Generating predictions on test data...")
    predictions = trained_model.transform(test_data)

    # Show sample predictions
    predictions.select("prediction", "label", "features").show(10)

    # Evaluate model accuracy
    evaluator = MulticlassClassificationEvaluator(
        labelCol='label',
        predictionCol='prediction',
        metricName='accuracy'
    )
    accuracy = evaluator.evaluate(predictions)
    print(f'Wine prediction model Test Accuracy = {accuracy}')

    # Evaluate model F1 Score
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol='label',
        predictionCol='prediction',
        metricName='f1'
    )
    f1_score = f1_evaluator.evaluate(predictions)
    print(f'Wine prediction model F1 Score = {f1_score}')

    spark.stop()



