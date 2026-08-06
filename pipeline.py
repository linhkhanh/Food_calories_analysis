import sqlite3
import os
import sys
import argparse
from functools import partial

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, explode, from_json, col
from pyspark.sql.types import StringType, StructType, StructField, FloatType, ArrayType
from service.model_service import process_single_image

# 1. Parse command line arguments passed from app.py
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="best_1", help="Model name to use")
args = parser.parse_args()
SELECTED_MODEL = args.model

# Initialize PySpark Session
spark = SparkSession.builder \
    .appName("FoodNutritionPipeline") \
    .config("spark.master", "local[*]") \
    .getOrCreate()

# 2. Bind SELECTED_MODEL to process_single_image using partial
func_with_model = partial(process_single_image, model_name=SELECTED_MODEL)

# Define PySpark UDF
run_ml_udf = udf(func_with_model, StringType())

# Read metadata CSV
df_input = spark.read.csv("data/input_metadata.csv", header=True, inferSchema=True)
df_processed = df_input.withColumn("ml_json", run_ml_udf(col("image_path"), col("plate_diameter_cm")))

# Updated PySpark Schema matching the new NUTRITION_LOOKUP attributes
schema = ArrayType(StructType([
    StructField("food_item", StringType()),
    StructField("estimated_weight_g", FloatType()),
    StructField("calories", FloatType()),
    StructField("protein_g", FloatType()),
    StructField("fat_g", FloatType()),
    StructField("carbs_g", FloatType()),
    StructField("sugars_g", FloatType()),
    StructField("cholesterol_mg", FloatType()),
    StructField("dietary_fiber_g", FloatType()),
    StructField("nutrition_density", FloatType()),
    StructField("ingredients", StringType()),
    StructField("annotated_path", StringType())
]))

# Parse JSON array returned by process_single_image
df_parsed = df_processed.withColumn("parsed", from_json(col("ml_json"), schema)) \
                        .select("image_id", "user_id", "timestamp", explode("parsed").alias("food")) \
                        .select(
                            "image_id", 
                            "user_id", 
                            "timestamp",
                            col("food.annotated_path").alias("image_path"),
                            col("food.food_item").alias("food_item"),
                            col("food.estimated_weight_g").alias("estimated_weight_g"),
                            col("food.calories").alias("calories"),
                            col("food.protein_g").alias("protein_g"),
                            col("food.fat_g").alias("fat_g"),
                            col("food.carbs_g").alias("carbs_g"),
                            col("food.sugars_g").alias("sugars_g"),
                            col("food.cholesterol_mg").alias("cholesterol_mg"),
                            col("food.dietary_fiber_g").alias("dietary_fiber_g"),
                            col("food.nutrition_density").alias("nutrition_density"),
                            col("food.ingredients").alias("ingredients")
                        )

def save_to_sqlite(target_df):
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect("database/nutrition_data.db")
    cursor = conn.cursor()
    
    # Updated SQLite Table Schema with new nutritional columns
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_nutrition (
            image_id TEXT, 
            user_id TEXT, 
            timestamp TEXT, 
            image_path TEXT, 
            food_item TEXT,
            estimated_weight_g REAL, 
            calories REAL, 
            protein_g REAL, 
            fat_g REAL, 
            carbs_g REAL,
            sugars_g REAL,
            cholesterol_mg REAL,
            dietary_fiber_g REAL,
            nutrition_density REAL,
            ingredients TEXT
        )
    ''')
    
    # Insert rows into SQLite
    for row in target_df.collect():
        cursor.execute('''
            INSERT INTO processed_nutrition VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            row['image_id'], 
            row['user_id'], 
            str(row['timestamp']), 
            row['image_path'],
            row['food_item'], 
            row['estimated_weight_g'], 
            row['calories'], 
            row['protein_g'], 
            row['fat_g'], 
            row['carbs_g'],
            row['sugars_g'],
            row['cholesterol_mg'],
            row['dietary_fiber_g'],
            row['nutrition_density'],
            row['ingredients']
        ))
    conn.commit()
    conn.close()

save_to_sqlite(df_parsed)

print("--- SUCCESS: BATCH PROCESSING COMPLETE WITH UPDATED NUTRITION LOOKUP FIELDS ---")
