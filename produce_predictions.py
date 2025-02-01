from joblib import load
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, StructType, StructField, FloatType
from pyspark.sql.functions import pandas_udf
import pandas as pd
import os

os.environ["JAVA_HOME"] = "/opt/homebrew/Cellar/openjdk@11/11.0.26/libexec/openjdk.jdk/Contents/Home"
os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-class-path /opt/homebrew/Cellar/openjdk@11/11.0.26/libexec/openjdk.jdk/Contents/Home pyspark-shell"

spark = SparkSession.builder \
    .appName("DiabetesIndicators") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.1") \
    .config("spark.driver.host", "192.168.0.23")\
    .getOrCreate()

model = load("best_model_XGB.joblib")
scaler = load("scaler.joblib")

columns = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income"
]

schema = StructType([StructField(column_name, DoubleType()) for column_name in columns])

data = spark \
    .readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "health_data") \
    .option("startingOffsets", "latest") \
    .load()

parsed_data = data.selectExpr("CAST(value AS STRING)") \
    .select(F.from_json(F.col("value"), schema).alias("data")) \
    .select("data.*")


@pandas_udf(FloatType())
def predict_udf(*cols):
    input_data = pd.concat(cols, axis=1)
    input_data.columns = columns

    scaled_data = scaler.transform(input_data)

    predictions = model.predict(scaled_data)

    return pd.Series(predictions)


predictions_df = parsed_data.withColumn("predicted_class", predict_udf(*parsed_data.columns))

output_df = predictions_df.select(
    F.to_json(F.struct("*")).alias("value")
)

query = output_df \
    .writeStream \
    .format("kafka") \
    .outputMode("append") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("topic", "health_data_predicted") \
    .option("checkpointLocation", "/tmp/checkpoint") \
    .start()

query.awaitTermination()