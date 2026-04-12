"""
Generate preprocessing step-by-step visual outputs + histograms for academic reporting.
Saves intermediate images and their pixel intensity histograms.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

REPORTS_DIR = "../reports"
PREPROCESS_DIR = os.path.join(REPORTS_DIR, "preprocessing_steps")
HISTOGRAM_DIR = os.path.join(REPORTS_DIR, "histograms")

def save_histogram(image, title, save_path):
    """Compute and save a pixel intensity histogram for a grayscale image."""
    plt.figure(figsize=(6, 4))
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(hist, color='black', linewidth=1.2)
    plt.fill_between(range(256), hist.flatten(), alpha=0.3, color='steelblue')
    plt.title(title, fontsize=13, fontweight='bold')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved histogram: {save_path}")

def process_single_image(img_path, img_index):
    """Run full preprocessing pipeline, saving each intermediate step + histograms."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"  Skipping unreadable: {img_path}")
        return
    
    basename = os.path.splitext(os.path.basename(img_path))[0]
    img_preprocess_dir = os.path.join(PREPROCESS_DIR, basename)
    img_histogram_dir = os.path.join(HISTOGRAM_DIR, basename)
    os.makedirs(img_preprocess_dir, exist_ok=True)
    os.makedirs(img_histogram_dir, exist_ok=True)
    
    # Step 0: Original
    cv2.imwrite(os.path.join(img_preprocess_dir, "0_original.png"), img)
    save_histogram(img, f"{basename} — Original", os.path.join(img_histogram_dir, "0_original_hist.png"))
    
    # Step 1: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(img_preprocess_dir, "1_grayscale.png"), gray)
    save_histogram(gray, f"{basename} — Grayscale", os.path.join(img_histogram_dir, "1_gray_hist.png"))
    
    # Step 2: Bilateral Filter
    blur = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    cv2.imwrite(os.path.join(img_preprocess_dir, "2_bilateral.png"), blur)
    save_histogram(blur, f"{basename} — Bilateral Filtered", os.path.join(img_histogram_dir, "2_bilateral_hist.png"))
    
    # Step 3: CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(blur)
    cv2.imwrite(os.path.join(img_preprocess_dir, "3_clahe.png"), contrast)
    save_histogram(contrast, f"{basename} — CLAHE", os.path.join(img_histogram_dir, "3_clahe_hist.png"))
    
    # Step 4: Adaptive Threshold
    thresh = cv2.adaptiveThreshold(
        contrast, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 11
    )
    cv2.imwrite(os.path.join(img_preprocess_dir, "4_threshold.png"), thresh)
    save_histogram(thresh, f"{basename} — Adaptive Threshold", os.path.join(img_histogram_dir, "4_thresh_hist.png"))
    
    # Step 5: Median Blur (Final Cleaned)
    cleaned = cv2.medianBlur(thresh, 3)
    out_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(img_preprocess_dir, "5_final.png"), out_bgr)
    save_histogram(cleaned, f"{basename} — Final Cleaned", os.path.join(img_histogram_dir, "5_final_hist.png"))
    
    # Step 6: Combined comparison strip
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    titles = ["Grayscale", "Bilateral", "CLAHE", "Threshold", "Final"]
    images = [gray, blur, contrast, thresh, cleaned]
    for ax, title, im in zip(axes, titles, images):
        ax.imshow(im, cmap='gray')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    plt.suptitle(f"Preprocessing Pipeline — {basename}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(img_preprocess_dir, "pipeline_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Completed: {basename}")

def main():
    os.makedirs(PREPROCESS_DIR, exist_ok=True)
    os.makedirs(HISTOGRAM_DIR, exist_ok=True)
    
    # Process sample images from training set
    sample_dirs = [
        "dataset/images/train",
        "dataset/images/val",
        "dataset/real_images",
    ]
    
    all_images = []
    for d in sample_dirs:
        if os.path.exists(d):
            files = glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.jpg"))
            all_images.extend(files)
    
    if not all_images:
        print("No images found. Checking dataset/images/ directly...")
        all_images = glob.glob("dataset/images/*.png")
    
    # Take a representative sample (max 10 images for report)
    sample = all_images[:10]
    
    print(f"\nGenerating preprocessing reports for {len(sample)} sample images...")
    print(f"Output: {os.path.abspath(PREPROCESS_DIR)}")
    print(f"Histograms: {os.path.abspath(HISTOGRAM_DIR)}\n")
    
    for idx, img_path in enumerate(sample):
        print(f"[{idx+1}/{len(sample)}] Processing: {os.path.basename(img_path)}")
        process_single_image(img_path, idx)
    
    print(f"\n{'='*60}")
    print("PREPROCESSING REPORT GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Step images: {os.path.abspath(PREPROCESS_DIR)}/")
    print(f"Histograms:  {os.path.abspath(HISTOGRAM_DIR)}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
