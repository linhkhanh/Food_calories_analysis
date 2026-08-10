### activate and deactivate virtual env

`source ~/tf-env/bin/activate`

To deactivate
`deactivate`

### run install

1. `pip install ultralytics pyspark streamlit pandas opencv-python-headless`

2. `pip install roboflow`
3. `pip install kaggle pandas`

### before running ( if ):

- Remove tempt images
- Remove annotated folder
- Remove nutrition_database.db

## Step 1:

### Create 1000 record csv file to test batch processing

`python helper/generate_csv.py`

## Step 2 (optional):

### Create SQLlite database

`python pipeline.py`

This step will create a annotation folder with images having bounding box on photo
Then create necessary table and insert data to db

## Step 3:

### To Run app

`streamlit run app.py`

If do this step without step 2, to test batch processing, please choose tab Batch Analytic Dashboard --> click Run Pyspark Batch pipeline Button -> wait for data Processing (it take around 1 - 3 mins)

### Models

There are 3 models to test in this project. Each model was trained with different dataset so the result of the analysis is different as well.
Trained model is as best.pt file, and located in weights folder.

To check how to apply model in code, please check function process_single_image in model_service.py

[ Input Image ]
│
▼
[ YOLOv8 Segmentation ] ───► Mask Contours & Food Class
│
├───────────────────► [ Reference Object Calibration ] ──► Real-World Area (cm²)
│ │
▼ ▼
[ Geometry Model ] ────────► [ Volume & Density Lookup ] ──────► Mass in Grams (g)
│
▼
[ USDA API / Macro Calculation ]
│
▼
[ Final UI & Analytics Output ]

### Dataset

1. https://universe.roboflow.com/designproject-180l3/food-segmentation-jel6c
2. https://universe.roboflow.com/hust-ajpvu/food-srnub
3. https://universe.roboflow.com/deepak-ojk8n/food-detection-17yv0

### Training Model

Training with Kaggle

```
# 1. Install ultralytics and roboflow
!pip install -q ultralytics roboflow

import os
from roboflow import Roboflow
from ultralytics import YOLO

# 2. Authenticate and download dataset (Roboflow automatically downloads to /kaggle/working/)
rf = Roboflow(api_key="Your API key")

# Download your food segmentation dataset
project = rf.workspace("designproject-180l3").project("food-segmentation-jel6c")
dataset = project.version(5).download("yolov8")

data_yaml_path = f"{dataset.location}/data.yaml"
print("Data YAML location:", data_yaml_path)
```

#### Run YOLOv8 Segmentation Training

```
from ultralytics import YOLO

# 1. Load pre-trained weights (yolov8m-seg or yolov8l-seg give higher accuracy for fine masks)
model = YOLO("yolov8m-seg.pt")

# 2. Train with optimized parameters
results = model.train(
    # --- Dataset & Workspace ---
    data=data_yaml_path,
    project="/kaggle/working",
    name="food_seg_optimized",
    save=True,

    # --- Core Training Setup ---
    epochs=100,                # 100 epochs gives the model time to converge
    patience=15,               # Early stopping: Stops training if no improvement for 15 epochs
    imgsz=640,                 # Standard high resolution for accurate mask edges
    batch=16,                  # Adjust down (e.g., 8) if you get CUDA OOM
    device=0,                  # Uses Kaggle primary GPU
    workers=8,                 # Multithreaded data loading (speeds up epoch transitions)

    # --- Optimizer & Learning Rate ---
    optimizer="AdamW",         # AdamW handles complex features & fine segmentation better than default SGD
    lr0=0.001,                 # Initial learning rate for AdamW
    lrf=0.01,                  # Final learning rate factor (decays down to lr0 * lrf)
    cos_lr=True,               # Cosine annealing learning rate schedule (smoother convergence)

    # --- Augmentations (Crucial for generalization) ---
    degrees=15.0,              # Small random rotations
    scale=0.5,                 # Scale zoom variations (0.5 means 50% to 150%)
    fliplr=0.5,                # Horizontal flip probability (50%)
    mosaic=1.0,                # Stitches 4 images into one (improves small object detection)

    # --- Performance Options ---
    plots=True                 # Auto-generates loss curves, confusion matrices, and sample images
)
```
