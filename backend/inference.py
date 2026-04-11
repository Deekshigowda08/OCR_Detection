import cv2
import numpy as np
from ultralytics import YOLO
from utils import preprocess_for_ocr, encode_image_base64

MODEL_PATH = "../runs/detect/train3/weights/best.pt"
model = None

try:
    # Load model once entirely globally (persists through multiple identical requests)
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Warning: Could not load YOLO model from {MODEL_PATH}. Error: {e}")

def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    xA = max(x1_min, x2_min)
    yA = max(y1_min, y2_min)
    xB = min(x1_max, x2_max)
    yB = min(y1_max, y2_max)

    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (x1_max - x1_min) * (y1_max - y1_min)
    box2Area = (x2_max - x2_min) * (y2_max - y2_min)
    unionArea = box1Area + box2Area - interArea
    
    if unionArea == 0:
        return 0
    return interArea / unionArea

def run_inference(img_bgr):
    if not model:
        return {"error": "Model not loaded"}
        
    preprocessed = preprocess_for_ocr(img_bgr)
    
    # Run primary inference step
    results = model(preprocessed, device="mps", verbose=False) 
    
    # Initial detection extraction
    boxes_data = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            char = model.names[cls_id]
            boxes_data.append({
                "box": [x1, y1, x2, y2],
                "conf": conf,
                "char": char
            })
            
    # Auto-Correction Algorithm (If Serial is upside down)
    if boxes_data:
        avg_conf = sum(b["conf"] for b in boxes_data) / len(boxes_data)
        if avg_conf < 0.6: 
            rotated = cv2.rotate(preprocessed, cv2.ROTATE_180)
            res_rot = model(rotated, device="mps", verbose=False)
            boxes_rot = []
            for r in res_rot:
                for box in r.boxes:
                    boxes_rot.append({
                        "box": box.xyxy[0].tolist(),
                        "conf": float(box.conf[0]),
                        "char": model.names[int(box.cls[0])]
                    })
            if boxes_rot:
                avg_conf_rot = sum(b["conf"] for b in boxes_rot) / len(boxes_rot)
                if avg_conf_rot > avg_conf:
                    # Switch completely to rotated approach logic
                    boxes_data = boxes_rot
                    preprocessed = rotated

    # Duplication filtering / Non-Max Suppression
    filtered_boxes = []
    # Prioritize highest confidence overlaps
    boxes_data.sort(key=lambda x: x["conf"], reverse=True)
    
    for bd in boxes_data:
        overlap = False
        for fbd in filtered_boxes:
            if compute_iou(bd["box"], fbd["box"]) > 0.4:
                overlap = True
                break
        if not overlap:
            filtered_boxes.append(bd)
            
    # Final String construction (Left-to-Right Sort)
    filtered_boxes.sort(key=lambda x: x["box"][0])
    
    detected_string = "".join([b["char"] for b in filtered_boxes])
    confs = [b["conf"] for b in filtered_boxes]
    ret_boxes = [b["box"] for b in filtered_boxes]
    
    # Graphic Generation Display Component
    annotated = preprocessed.copy()
    for bd in filtered_boxes:
        x1, y1, x2, y2 = map(int, bd["box"])
        char = bd["char"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, char, (x1, max(y1-10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
    base64_out = encode_image_base64(annotated)
    preprocessed_base64 = encode_image_base64(preprocessed)
    
    return {
        "text": detected_string,
        "boxes": ret_boxes,
        "confidences": confs,
        "annotated_image": f"data:image/png;base64,{base64_out}",
        "preprocessed_image": f"data:image/png;base64,{preprocessed_base64}"
    }
