### activate and deactivate virtual env

`source ~/tf-env/bin/activate`
`deactivate`

### run install

`pip install ultralytics pyspark streamlit pandas opencv-python-headless`
`pip install roboflow`
`pip install kaggle pandas`

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
