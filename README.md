Command line:

### activate virtual env

source ~/tf-env/bin/activate
deactivate

### run install

pip install ultralytics pyspark streamlit pandas opencv-python-headless
pip install roboflow
pip install kaggle pandas

### before running:

- Remove tempt images
- Remove annotated folder
- Remove nutrition_database.db

### To create 1000 record csv file

python helper/generate_csv.py

### To Create SQLlite database

draw bounding box on photo

python pipeline.py

### To Run app

streamlit run app.py

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

## Install interference

The error occurs because the inference library does not support Python 3.13

### Install Python 3.12

brew install python@3.12

### Create a Python 3.12 Virtual Environment

python3.12 -m venv inference-env

### Activate the Environment:

source inference-env/bin/activate

### Install the Library

pip install --upgrade pip
pip install inference
