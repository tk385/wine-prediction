import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.sql import DataFrame
import boto3

def clean_data(df: DataFrame) -> DataFrame:
    """Clean and prepare DataFrame by casting all columns to double."""
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))

def save_single_file(dataframe: DataFrame, output_path: str):
    """Save the DataFrame as a single CSV file in S3."""
    temp_path = "s3://wineprediction3/datasets/temp_test_data"
    dataframe.coalesce(1).write.mode("overwrite").csv(temp_path, header=True, sep=";")

    s3 = boto3.resource('s3')
    bucket_name = "wineprediction3"
    temp_folder = "datasets/temp_test_data"
    target_path = "datasets/TestDataset.csv"

    bucket = s3.Bucket(bucket_name)
    part_file = next((obj.key for obj in bucket.objects.filter(Prefix=temp_folder) if obj.key.endswith(".csv")), None)

    if part_file:
        bucket.copy({'Bucket': bucket_name, 'Key': part_file}, target_path)
        bucket.objects.filter(Prefix=temp_folder).delete()

if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName('Wine_Quality_Prediction_RF') \
        .config("spark.sql.shuffle.partitions", "50") \
        .getOrCreate()

    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

    # Paths
    input_path = sys.argv[1] if len(sys.argv) > 1 else "s3://wineprediction3/datasets/TrainingDataset.csv"
    valid_path = sys.argv[2] if len(sys.argv) > 2 else "s3://wineprediction3/datasets/ValidationDataset.csv"
    model_output_path = sys.argv[3] if len(sys.argv) > 3 else "s3://wineprediction3/models/wine_quality_model"

    print("Loading and cleaning training dataset...")
    train_df = spark.read.format("csv").option('header', 'true').option("sep", ";").load(input_path)
    cleaned_train_df = clean_data(train_df)

    print("Splitting the dataset into training and testing...")
    train_data, test_data = cleaned_train_df.randomSplit([0.9, 0.1], seed=42)

    print("Saving test dataset to S3 as a single file...")
    save_single_file(test_data, "s3://wineprediction3/datasets/TestDataset.csv")

    print("Loading and cleaning validation dataset...")
    valid_df = spark.read.format("csv").option('header', 'true').option("sep", ";").load(valid_path)
    valid_data = clean_data(valid_df)

    # Features and label
    all_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                    'pH', 'sulphates', 'alcohol']

    assembler = VectorAssembler(inputCols=all_features, outputCol='features')
    indexer = StringIndexer(inputCol="quality", outputCol="label")
    rf = RandomForestClassifier(labelCol='label', featuresCol='features', seed=42)

    # Pipeline
    pipeline = Pipeline(stages=[assembler, indexer, rf])

    # Hyperparameter tuning
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [200, 500]) \
        .addGrid(rf.maxDepth, [10, 20]) \
        .addGrid(rf.minInstancesPerNode, [1, 5]) \
        .build()

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    crossval = CrossValidator(estimator=pipeline, 
                              estimatorParamMaps=paramGrid, 
                              evaluator=evaluator, 
                              numFolds=3, 
                              parallelism=4)

    print("Training with Cross-Validation...")
    cv_model = crossval.fit(train_data)

    print("Evaluating the model on validation data...")
    predictions = cv_model.bestModel.transform(valid_data)
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    f1_score = evaluator.evaluate(predictions)

    print(f"Best Model Test Accuracy: {accuracy}")
    print(f"Best Model Weighted F1 Score: {f1_score}")

    print(f"Saving the best model to {model_output_path}...")
    cv_model.bestModel.write().overwrite().save(model_output_path)

    spark.stop()
