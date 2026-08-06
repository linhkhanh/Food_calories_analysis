import json
import os
import cv2
import numpy as np
from ultralytics import YOLO
# from service.nutrition_lookup import NUTRITION_LOOKUP
from service.mock_data import NUTRITION_LOOKUP

ANNOTATED_DIR = "data/annotated"
os.makedirs(ANNOTATED_DIR, exist_ok=True)

model = YOLO("weights/best_5.pt")

def polygon_area(coords):
    x, y = coords[:, 0], coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def process_single_image(image_path, plate_cm=26.0):
    try:
        results = model.predict(source=image_path, conf=0.25, verbose=False)[0]
        
        # 1. Vẽ Bounding Boxes / Masks lên ảnh và lưu file mới
        annotated_frame = results.plot()
        base_name = os.path.basename(image_path)
        annotated_path = os.path.join(ANNOTATED_DIR, f"annotated_{base_name}")
        cv2.imwrite(annotated_path, annotated_frame)
        
        output_items = []
        plate_cm = float(plate_cm) if plate_cm else 26.0
        
        # 2. Bóc tách thông tin nhận diện
        if results.masks is not None:
            for mask, box in zip(results.masks.xy, results.boxes):
                cls_id = int(box.cls[0].item())
                food_name = model.names[cls_id]
                
                polygon = np.array(mask, dtype=np.int32)
                pixel_area = float(polygon_area(polygon))
                
                scale_factor = (plate_cm / 500.0) ** 2
                area_cm2 = pixel_area * scale_factor
                volume_cm3 = area_cm2 * 2.0
                
                info = NUTRITION_LOOKUP.get(food_name.lower(), NUTRITION_LOOKUP["default"])
                # Format ingredients: convert list to comma-separated string if needed
                ingredients_val = info.get("ingredients", [])
                if isinstance(ingredients_val, list):
                    ingredients_str = ", ".join(ingredients_val)
                else:
                    ingredients_str = str(ingredients_val)

                weight_g = volume_cm3 * info["density"]
                ratio = weight_g / 100.0
                
                output_items.append({
                    "food_item": food_name,
                    "estimated_weight_g": round(weight_g, 1),
                    "calories": round(ratio * info.get("cal_100g", 0.0), 1),
                    "protein_g": round(ratio * info.get("protein", 0.0), 1),
                    "fat_g": round(ratio * info.get("fat", 0.0), 1),
                    "carbs_g": round(ratio * info.get("carbs", 0.0), 1),
                    "sugars_g": round(ratio * info.get("sugars", 0.0), 1),
                    "cholesterol_mg": round(ratio * info.get("cholesterol", 0.0), 1),
                    "dietary_fiber_g": round(ratio * info.get("dietary_fiber", 0.0), 1),
                    "nutrition_density": round(float(info.get("nutrition_density", 0.0)), 1),
                    "ingredients": ingredients_str,
                    "annotated_path": annotated_path  # Trả về đường dẫn ảnh đã vẽ Box
                })
        return json.dumps(output_items)
    except Exception as e:
        return json.dumps([])
    