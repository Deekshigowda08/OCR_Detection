"""
Prepare combined dataset for YOLO fine-tuning.
Merges real-world pseudo-labeled images with existing synthetic data,
shuffles, and splits into train/val (80/20).
Works on both Windows and macOS.
"""
import os
import shutil
import random
import glob

# ============================================================
# CONFIGURATION — Edit these paths if your layout is different
# ============================================================
REAL_IMAGES_DIR = os.path.join("dataset", "real_images_processed")
REAL_LABELS_DIR = os.path.join("dataset", "real_labels")

SYNTHETIC_IMAGES_DIR = os.path.join("dataset", "images")
SYNTHETIC_LABELS_DIR = os.path.join("dataset", "labels")

OUTPUT_IMAGES_TRAIN = os.path.join("dataset", "images", "train")
OUTPUT_IMAGES_VAL   = os.path.join("dataset", "images", "val")
OUTPUT_LABELS_TRAIN = os.path.join("dataset", "labels", "train")
OUTPUT_LABELS_VAL   = os.path.join("dataset", "labels", "val")

SPLIT_RATIO = 0.8  # 80% train, 20% val
RANDOM_SEED = 42

def collect_pairs(images_dir, labels_dir):
    """Collect matched (image, label) pairs from directories."""
    pairs = []
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        return pairs
    
    valid_exts = ('.png', '.jpg', '.jpeg', '.webp')
    for filename in os.listdir(images_dir):
        if filename.lower().endswith(valid_exts):
            basename = os.path.splitext(filename)[0]
            label_path = os.path.join(labels_dir, basename + ".txt")
            image_path = os.path.join(images_dir, filename)
            if os.path.exists(label_path):
                pairs.append((image_path, label_path, filename))
    return pairs

def main():
    # Create output directories
    for d in [OUTPUT_IMAGES_TRAIN, OUTPUT_IMAGES_VAL, OUTPUT_LABELS_TRAIN, OUTPUT_LABELS_VAL]:
        os.makedirs(d, exist_ok=True)
    
    all_pairs = []
    
    # 1. Collect real-world labeled pairs
    real_pairs = collect_pairs(REAL_IMAGES_DIR, REAL_LABELS_DIR)
    print(f"Real-world image-label pairs found: {len(real_pairs)}")
    all_pairs.extend(real_pairs)
    
    # 2. Collect existing synthetic pairs (from flat dataset/images + dataset/labels)
    #    Only if they aren't already in train/val subdirectories
    synth_pairs = []
    if os.path.exists(SYNTHETIC_IMAGES_DIR) and os.path.exists(SYNTHETIC_LABELS_DIR):
        for filename in os.listdir(SYNTHETIC_IMAGES_DIR):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(SYNTHETIC_IMAGES_DIR, filename)
                # Skip if it's actually a directory (train/val)
                if os.path.isdir(img_path):
                    continue
                basename = os.path.splitext(filename)[0]
                lbl_path = os.path.join(SYNTHETIC_LABELS_DIR, basename + ".txt")
                if os.path.exists(lbl_path):
                    synth_pairs.append((img_path, lbl_path, filename))
    print(f"Synthetic image-label pairs found: {len(synth_pairs)}")
    all_pairs.extend(synth_pairs)
    
    # 3. Also collect any existing train/val pairs
    existing_train = collect_pairs(OUTPUT_IMAGES_TRAIN, OUTPUT_LABELS_TRAIN)
    existing_val = collect_pairs(OUTPUT_IMAGES_VAL, OUTPUT_LABELS_VAL)
    print(f"Existing train pairs: {len(existing_train)}")
    print(f"Existing val pairs: {len(existing_val)}")
    
    if not all_pairs and not existing_train:
        print("\nERROR: No image-label pairs found!")
        print(f"Make sure images are in: {REAL_IMAGES_DIR}")
        print(f"And labels are in: {REAL_LABELS_DIR}")
        print("Run auto_label.py first to generate labels.")
        return
    
    if not all_pairs:
        print("\nNo new pairs to add. Existing dataset is ready.")
        return
    
    # 4. Shuffle and split
    random.seed(RANDOM_SEED)
    random.shuffle(all_pairs)
    
    split_idx = int(len(all_pairs) * SPLIT_RATIO)
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    
    print(f"\nTotal new pairs: {len(all_pairs)}")
    print(f"Train split: {len(train_pairs)}")
    print(f"Val split: {len(val_pairs)}")
    
    # 5. Copy files
    copied = 0
    for pairs, img_dir, lbl_dir in [
        (train_pairs, OUTPUT_IMAGES_TRAIN, OUTPUT_LABELS_TRAIN),
        (val_pairs, OUTPUT_IMAGES_VAL, OUTPUT_LABELS_VAL)
    ]:
        for img_path, lbl_path, filename in pairs:
            basename = os.path.splitext(filename)[0]
            
            # Add prefix to avoid collisions between real and synthetic
            dst_img = os.path.join(img_dir, filename)
            dst_lbl = os.path.join(lbl_dir, basename + ".txt")
            
            # Skip if source and destination are the same file
            if os.path.abspath(img_path) == os.path.abspath(dst_img):
                continue
            
            shutil.copy2(img_path, dst_img)
            shutil.copy2(lbl_path, dst_lbl)
            copied += 1
    
    # Count final totals
    train_count = len([f for f in os.listdir(OUTPUT_IMAGES_TRAIN) if not os.path.isdir(os.path.join(OUTPUT_IMAGES_TRAIN, f))])
    val_count = len([f for f in os.listdir(OUTPUT_IMAGES_VAL) if not os.path.isdir(os.path.join(OUTPUT_IMAGES_VAL, f))])
    
    print(f"\n{'='*50}")
    print("DATASET PREPARATION COMPLETE")
    print(f"{'='*50}")
    print(f"Copied {copied} new files")
    print(f"Final train images: {train_count}")
    print(f"Final val images:   {val_count}")
    print(f"Train dir: {os.path.abspath(OUTPUT_IMAGES_TRAIN)}")
    print(f"Val dir:   {os.path.abspath(OUTPUT_IMAGES_VAL)}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
