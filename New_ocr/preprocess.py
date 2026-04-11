import cv2
import numpy as np
import os
import glob

def preprocess(image_input):
    """
    Preprocess image for OCR/YOLO (Optimized for Industrial Metal/Rust).
    image_input: can be a file path string or a numpy array (BGR/RGB).
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_input}")
    else:
        img = image_input

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 1. Bilateral Filtering
    # Crucial for metallic surfaces: Blurs out rust/scratches while keeping character edges razor sharp
    blur = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Neutralizes extreme specular highlights (glare) and deep shadows on curved metal
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(blur)
    
    # 3. Adaptive Gaussian Thresholding
    # Otsu fails on unevenly lit metal. Adaptive isolation calculates thresholds locally per 21x21 block
    thresh = cv2.adaptiveThreshold(
        contrast, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 11
    )
    
    # 4. Salt-and-Pepper Noise Removal
    # Median blurring mathematically removes the isolated speckles caused by rust under adaptive thresholding
    cleaned = cv2.medianBlur(thresh, 3)
    
    # Reconvert back to BGR 3-channel for YOLO compatibility
    out_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

    return out_bgr


if __name__ == "__main__":
    # Process the entire dataset and overwrite
    test_dir = "dataset/images"
    
    image_files = glob.glob(os.path.join(test_dir, "*.png"))
    
    if not image_files:
        print(f"No images found in {test_dir}. Run gen_data.py first!")
    else:
        print(f"Starting mass preprocessing of {len(image_files)} images (In-place)...")
        for idx, file_path in enumerate(image_files):
            orig = cv2.imread(file_path)
            
            # Apply preprocessing
            processed = preprocess(orig)
            
            # Overwrite original
            cv2.imwrite(file_path, processed)
            
            if (idx + 1) % 500 == 0:
                print(f"Processed {idx + 1}/{len(image_files)} images...")
                
        print("Done! All images have been preprocessed in place.")
