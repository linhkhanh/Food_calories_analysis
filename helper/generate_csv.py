import os
import csv
import random
from datetime import datetime, timedelta

IMAGE_DIR = "data/images"
CSV_PATH = "data/input_metadata.csv"
TARGET_COUNT = 1000

# Scan folder for image extensions
supported_exts = (".png", ".jpg", ".jpeg")

if os.path.exists(IMAGE_DIR):
    available_images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(supported_exts)]
else:
    available_images = []

if not available_images:
    print(f"⚠️ No images found in '{IMAGE_DIR}'. Using dummy filenames for testing.")
    available_images = [f"sample_food_{i}.jpg" for i in range(1, 11)]

# Ensure output directory exists
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

base_time = datetime.now() - timedelta(days=30)

with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # Write CSV Header required by PySpark pipeline
    writer.writerow(["image_id", "user_id", "timestamp", "image_path", "plate_diameter_cm"])
    
    for i in range(1, TARGET_COUNT + 1):
        image_id = f"IMG_{i:05d}"
        
        # Distribute across 50 simulated users
        user_id = f"USER_{(i % 50) + 100}"
        
        # Spread timestamps over the last 30 days to simulate real log data
        random_minutes = random.randint(0, 30 * 24 * 60)
        timestamp = (base_time + timedelta(minutes=random_minutes)).strftime("%Y-%m-%d %H:%M:%S")
        
        # Cycle through available local images
        filename = available_images[(i - 1) % len(available_images)]
        full_path = os.path.join(IMAGE_DIR, filename)
        
        # Slightly vary plate diameter between 24.0 cm and 28.0 cm for data diversity
        plate_diameter = round(random.uniform(24.0, 28.0), 1)
        
        writer.writerow([image_id, user_id, timestamp, full_path, plate_diameter])

print(f"✅ Successfully created '{CSV_PATH}' with {TARGET_COUNT} records!")
