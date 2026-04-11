import base64
import io
import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import os

def encode_image_base64(img_bgr):
    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')

def decode_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def preprocess_for_ocr(img):
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Bilateral Filtering
    # Crucial for metallic surfaces: Blurs out rust/scratches while keeping character edges razor sharp
    blur = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Neutralizes extreme specular highlights (glare) and deep shadows on curved metal
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(blur)
    
    # 4. Adaptive Gaussian Thresholding
    # Otsu fails on unevenly lit metal. Adaptive isolation calculates thresholds locally per 21x21 block
    thresh = cv2.adaptiveThreshold(
        contrast, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 11
    )
    
    # 5. Salt-and-Pepper Noise Removal
    # Median blurring mathematically removes the isolated speckles caused by rust under adaptive thresholding
    cleaned = cv2.medianBlur(thresh, 3)
    
    # Reconvert back to BGR 3-channel for YOLO ingestion
    out_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    return out_bgr

def generate_metrics_graphs():
    csv_path = "../runs/detect/train3/results.csv"
    if not os.path.exists(csv_path):
        return None

    epochs = []
    train_loss = []
    map50 = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Parse headers flexibly to handle versioning changes
        header = [h.strip() for h in header]
        try:
            ep_idx = header.index("epoch")
            loss_idx = header.index("train/cls_loss")
            map_idx = header.index("metrics/mAP50(B)")
        except ValueError:
            ep_idx, loss_idx, map_idx = 0, 2, 6 # Common fallback structure

        for row in reader:
            # Skip empty or malformed rows during live training
            if len(row) > max(ep_idx, loss_idx, map_idx):
                epochs.append(float(row[ep_idx]))
                train_loss.append(float(row[loss_idx]))
                map50.append(float(row[map_idx]))

    # Plot Training Loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label='Train Class Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    buf_loss = io.BytesIO()
    plt.savefig(buf_loss, format='png', bbox_inches='tight')
    plt.close()
    
    # Plot Validation mAP
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, map50, label='mAP@0.5', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.title('mAP vs Epochs')
    plt.grid(True)
    buf_map = io.BytesIO()
    plt.savefig(buf_map, format='png', bbox_inches='tight')
    plt.close()

    loss_b64 = base64.b64encode(buf_loss.getvalue()).decode('utf-8')
    map_b64 = base64.b64encode(buf_map.getvalue()).decode('utf-8')

    def encode_file_base64(filepath):
        if not os.path.exists(filepath):
            return None
        with open(filepath, "rb") as file_obj:
            return base64.b64encode(file_obj.read()).decode('utf-8')
            
    conf_matrix_b64 = encode_file_base64("../runs/detect/train3/confusion_matrix.png")
    pr_curve_b64 = encode_file_base64("../runs/detect/train3/PR_curve.png")
    f1_curve_b64 = encode_file_base64("../runs/detect/train3/F1_curve.png")
    
    return {
        "loss_graph": f"data:image/png;base64,{loss_b64}", 
        "map_graph": f"data:image/png;base64,{map_b64}",
        "confusion_matrix": f"data:image/png;base64,{conf_matrix_b64}" if conf_matrix_b64 else None,
        "pr_curve": f"data:image/png;base64,{pr_curve_b64}" if pr_curve_b64 else None,
        "f1_curve": f"data:image/png;base64,{f1_curve_b64}" if f1_curve_b64 else None
    }
