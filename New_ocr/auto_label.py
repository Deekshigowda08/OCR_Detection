import os
import sys
import cv2
from ultralytics import YOLO

# Add parent directory to path to safely import from backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from backend.utils import preprocess_for_ocr as preprocess
except ImportError:
    # Alternative fallback if run directly from root
    from New_ocr.preprocess import preprocess

def auto_label_images(unlabelled_dir, output_labels_dir, processed_images_dir, model_path=None):
    """
    Passes raw, unlabelled real-world images through your current YOLO model, 
    generating YOLO-format .txt files automatically to save you from drawing boxes from scratch.
    """
    if model_path is None:
        model_path = os.path.join("..", "runs", "detect", "train3", "weights", "best.pt")
    
    CONF_THRESH = 0.5
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Auto-detect device
    import torch
    if torch.cuda.is_available():
        device = "0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    os.makedirs(output_labels_dir, exist_ok=True)
    os.makedirs(processed_images_dir, exist_ok=True)
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    
    processed_count = 0
    skipped_count = 0
    
    for filename in os.listdir(unlabelled_dir):
        if filename.lower().endswith(valid_exts):
            img_path = os.path.join(unlabelled_dir, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            # Preprocess the image to normalize exactly as the training pipeline did
            processed_img = preprocess(img)
            proc_h, proc_w = processed_img.shape[:2]
                
            # Run inference
            results = model(processed_img, device=device, verbose=False)
            
            # Extract and filter target detections
            detections = []
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf >= CONF_THRESH:
                        cls_id = int(box.cls[0])
                        x_center, y_center, width, height = box.xywhn[0].tolist()
                        x_pixel = x_center * proc_w
                        
                        detections.append({
                            "cls_id": cls_id,
                            "x_center": x_center,
                            "y_center": y_center,
                            "width": width,
                            "height": height,
                            "conf": conf,
                            "x_pixel": x_pixel
                        })
            
            if not detections:
                skipped_count += 1
                print(f"Skipped {filename}: 0 detections met threshold >= {CONF_THRESH}")
                continue
                
            # Sort detections Left to Right by x_center
            detections.sort(key=lambda d: d["x_center"])
            
            # NMS filter: Remove duplicate / overlapping boxes (robust normalized 0.01 threshold)
            filtered_detections = []
            for d in detections:
                if not filtered_detections:
                    filtered_detections.append(d)
                else:
                    last_d = filtered_detections[-1]
                    if abs(d["x_center"] - last_d["x_center"]) < 0.01:
                        # Keep higher confidence
                        if d["conf"] > last_d["conf"]:
                            filtered_detections[-1] = d
                    else:
                        filtered_detections.append(d)
            
            # Prepare txt file path
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(output_labels_dir, txt_filename)
            
            with open(txt_path, 'w') as f:
                for d in filtered_detections:
                    f.write(f"{d['cls_id']} {d['x_center']:.6f} {d['y_center']:.6f} {d['width']:.6f} {d['height']:.6f}\n")
            
            # Write the exact processed image alongside the labels so data alignment is flawless
            processed_img_path = os.path.join(processed_images_dir, filename)
            cv2.imwrite(processed_img_path, processed_img)
                        
            processed_count += 1
            print(f"Pseudo-labeled: {filename} -> Recorded {len(filtered_detections)} distinct characters")
            
    print(f"\n✅ Total images successfully labeled: {processed_count}")
    print(f"❌ Skipped (No text / Empty): {skipped_count}")
            
    print(f"\n✅ Successfully auto-labeled {processed_count} images.")
    print(f"You can now load these into an annotator like LabelImg or Roboflow to manually fix the 30% that failed, rather than drawing 100% from scratch!")

if __name__ == "__main__":
    # 1. Define where your newly downloaded real-world images are located
    REAL_WORLD_IMAGES_DIR = "dataset/real_images"
    
    # 2. Define where the script should spit out the YOLO .txt label files
    GENERATED_LABELS_DIR = "dataset/real_labels"
    
    # 3. Define where to safely dump the modified preprocessed images!
    PROCESSED_IMAGES_DIR = "dataset/real_images_processed"
    
    # Make sure your input directory exists
    os.makedirs(REAL_WORLD_IMAGES_DIR, exist_ok=True)
    
    # Run the auto-labeler
    auto_label_images(REAL_WORLD_IMAGES_DIR, GENERATED_LABELS_DIR, PROCESSED_IMAGES_DIR)
