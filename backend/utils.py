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
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE Handle uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Light Gaussian blur for noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    # Reconvert back to BGR 3-channel for standard YOLO ingestion
    out_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
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
    
    return {
        "loss_graph": f"data:image/png;base64,{loss_b64}", 
        "map_graph": f"data:image/png;base64,{map_b64}"
    }
