import cv2
import numpy as np
import os
import glob

def preprocess(image_input):
    """
    Preprocess image for OCR/YOLO.
    image_input: can be a file path string or a numpy array (BGR/RGB).
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_input}")
    else:
        img = image_input

    # 2a) Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 2b) Contrast enhancement (CLAHE)
    # clipLimit ~ 2.0, tileGridSize=(8, 8) handles uneven lighting well
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2c) Noise reduction
    # Light Gaussian blur mitigates noise before sharpening
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # 2d) Sharpening
    # Strongly enhances character edges
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(blurred, -1, sharpen_kernel)

    # 2e) Adaptive / Otsu Thresholding
    # Since text is bright on dark backgrounds in our dataset, Otsu will naturally handle this.
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2f) Morphological operations
    # Closing bridges small gaps and repairs broken characters commonly caused by noise and thresholding
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, morph_kernel)

    # 3) Convert back to 3-channel (BGR) for YOLO compatibility
    out_bgr = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)

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
