import streamlit as st
import pandas as pd
import sqlite3
import os
import csv
import subprocess
import json
from datetime import datetime

# Import model logic directly for instant processing
from service.model_service import process_single_image

st.set_page_config(page_title="AI Food & Calorie Dashboard", layout="wide")
st.title("🥗 Big Data Food Nutrition Analytics")

IMAGE_DIR = "data/images"
CSV_PATH = "data/input_metadata.csv"
DB_PATH = "database/nutrition_data.db"
os.makedirs(IMAGE_DIR, exist_ok=True)

# Define display columns matching updated NUTRITION_LOOKUP fields
DISPLAY_COLUMNS = [
    'food_item',
    'estimated_weight_g',
    'calories',
    'protein_g',
    'fat_g',
    'carbs_g',
    'sugars_g',
    'cholesterol_mg',
    'dietary_fiber_g',
    'nutrition_density',
    'ingredients'
]

# Create Tabs for Navigation
tab_instant, tab_batch = st.tabs(["⚡ Instant Photo Analyzer", "📊 Batch Analytics Dashboard"])

# ==========================================
# TAB 1: INSTANT PHOTO ANALYZER
# ==========================================
with tab_instant:
    st.header("⚡ Instant Meal Analysis")
    st.write("Upload one or multiple photos to get immediate bounding boxes and nutritional breakdown.")

    col_up, col_param = st.columns([2, 1])
    with col_param:
        plate_cm_instant = st.number_input("Plate Diameter (cm)", min_value=10.0, max_value=50.0, value=26.0, key="instant_plate")
    
    with col_up:
        uploaded_files = st.file_uploader(
            "Choose meal photo(s)...", 
            type=["jpg", "png", "jpeg"], 
            accept_multiple_files=True,
            key="instant_files"
        )

    if uploaded_files:
        if st.button("🔍 Analyze Uploaded Photos", type="primary"):
            st.divider()
            
            for file in uploaded_files:
                # 1. Save uploaded file temporarily
                temp_filename = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.name}"
                temp_path = os.path.join(IMAGE_DIR, temp_filename)
                
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # 2. Run Instant Inference using model_service
                with st.spinner(f"Analyzing {file.name}..."):
                    json_res = process_single_image(temp_path, plate_cm=plate_cm_instant)
                    detected_items = json.loads(json_res)
                
                # 3. Render Results Container
                with st.container():
                    st.subheader(f"📸 Results for: `{file.name}`")
                    c1, c2 = st.columns([1, 2])
                    
                    if detected_items:
                        annotated_path = detected_items[0].get("annotated_path", temp_path)
                        
                        with c1:
                            if os.path.exists(annotated_path):
                                st.image(annotated_path, caption=f"Detected: {file.name}", use_container_width=True)
                            else:
                                st.image(temp_path, caption="Original Image", use_container_width=True)
                                
                        with c2:
                            df_res = pd.DataFrame(detected_items)
                            
                            # Standardize column existence in case keys are missing
                            for col_name in DISPLAY_COLUMNS:
                                if col_name not in df_res.columns:
                                    df_res[col_name] = 0.0
                            st.write("**Detected Food Items Breakdown:**")
                            st.dataframe(
                                df_res[DISPLAY_COLUMNS],
                                use_container_width=True,
                                hide_index=True
                            )
                    else:
                        with c1:
                            st.image(temp_path, caption="No objects detected", use_container_width=True)
                        with c2:
                            st.warning("No food items detected in this image. Try adjusting confidence or uploading another image.")
                
                st.divider()

# ==========================================
# TAB 2: BATCH ANALYTICS DASHBOARD
# ==========================================
with tab_batch:
    st.header("📊 Batch Processing & Long-Term Analytics")
    
    # --- SIDEBAR / BATCH CONTROLS ---
    st.sidebar.header("📤 Add to Batch Pipeline")
    with st.sidebar.form("upload_form"):
        user_id = st.text_input("User ID", value="USER_101")
        plate_cm = st.number_input("Plate Diameter (cm)", min_value=10.0, max_value=50.0, value=26.0, key="batch_plate")
        uploaded_file = st.file_uploader("Choose a food image for Batch...", type=["jpg", "png", "jpeg"], key="batch_file")
        submit = st.form_submit_button("Upload & Save Metadata")

    if submit and uploaded_file is not None:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_id = f"IMG_{timestamp_str}"
        saved_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
        
        with open(saved_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        file_exists = os.path.exists(CSV_PATH)
        with open(CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["image_id", "user_id", "timestamp", "image_path", "plate_diameter_cm"])
            writer.writerow([image_id, user_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), saved_path, plate_cm])
            
        st.sidebar.success("Image uploaded & appended to input_metadata.csv!")

    if st.sidebar.button("⚡ Run PySpark Batch Pipeline"):
        with st.spinner("Processing batch data with PySpark..."):
            start_time = datetime.now()
            subprocess.run(["python", "pipeline.py"])
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            st.success("PySpark execution completed!. Time taken: {:.2f} seconds".format(total_time))

    # --- RENDER BATCH DATABASE RESULTS ---
    if os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM processed_nutrition", conn)
        conn.close()

        if not df.empty:
            # Standardize column existence for historical records
            for col_name in DISPLAY_COLUMNS:
                if col_name not in df.columns:
                    df[col_name] = 0.0

            # Overview Cards Row 1
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Meals Processed", df['image_id'].nunique())
            col2.metric("Total Calories (kcal)", f"{df['calories'].sum():,.1f}")
            col3.metric("Total Protein (g)", f"{df['protein_g'].sum():,.1f}")
            col4.metric("Total Fat (g)", f"{df['fat_g'].sum():,.1f}")

            # Overview Cards Row 2
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Total Sugars (g)", f"{df['sugars_g'].sum():,.1f}")
            col6.metric("Total Cholesterol (mg)", f"{df['cholesterol_mg'].sum():,.1f}")
            col7.metric("Total Fiber (g)", f"{df['dietary_fiber_g'].sum():,.1f}")
            col8.metric("Overall Avg Nutrition Density", f"{df['nutrition_density'].mean():.1f}")

            st.markdown("---")
            st.write("### 🖼️ Historic Processed Meals Gallery")

            # 1. Initialize session state for batch limit
            BATCH_SIZE = 10  # Number of images to display per batch

            if "visible_image_count" not in st.session_state:
                st.session_state.visible_image_count = BATCH_SIZE

            # 2. Get unique images DataFrame
            unique_images = df[['image_id', 'image_path', 'user_id', 'timestamp']].drop_duplicates()
            total_images = len(unique_images)

            # 3. Slice the DataFrame to render only the visible subset
            visible_images = unique_images.head(st.session_state.visible_image_count)

            for _, img_row in visible_images.iterrows():
                img_id = img_row['image_id']
                img_path = img_row['image_path']
                items_df = df[df['image_id'] == img_id]
                
                with st.container():
                    col_img, col_details = st.columns([1, 2])
                    
                    with col_img:
                        if img_path and os.path.exists(img_path):
                            st.image(img_path, caption=f"Detected: {img_id} ({img_row['user_id']})", use_container_width=True)
                        else:
                            st.warning(f"Image not found at `{img_path}`")
                    
                    with col_details:
                        st.markdown(f"**Timestamp:** `{img_row['timestamp']}`")
                        st.markdown(f"**Total Meal Calories:** `{items_df['calories'].sum():.1f} kcal` | **Avg Density:** `{items_df['nutrition_density'].mean():.1f}`")
                        
                        st.dataframe(
                            items_df[DISPLAY_COLUMNS],
                            use_container_width=True,
                            hide_index=True
                        )
                    st.divider()

            # 4. "Load More" Button logic
            if st.session_state.visible_image_count < total_images:
                remaining_count = total_images - st.session_state.visible_image_count
                
                if st.button(f"Load More ({min(BATCH_SIZE, remaining_count)} images), remaining: {remaining_count}"):
                    st.session_state.visible_image_count += BATCH_SIZE
                    st.rerun()

            # 5. Charts / Analytics (Runs after or alongside the list)
            st.write("### Calorie Distribution by Food Item")
            st.bar_chart(df.groupby('food_item')['calories'].sum())
        else:
            st.info("Database is empty. Run the PySpark pipeline to populate data.")
    else:
        st.info("No database found yet. Upload an image and click 'Run PySpark Batch Pipeline'.")
