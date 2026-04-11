import os
import random
import shutil
import glob

def setup_directories(base_dir):
    # Dirs
    dirs = [
        os.path.join(base_dir, 'images', 'train'),
        os.path.join(base_dir, 'images', 'val'),
        os.path.join(base_dir, 'labels', 'train'),
        os.path.join(base_dir, 'labels', 'val')
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs

def main():
    base_dir = "dataset"
    img_dir = os.path.join(base_dir, "images")
    lbl_dir = os.path.join(base_dir, "labels")
    
    # 1. Setup nested training directories
    img_train, img_val, lbl_train, lbl_val = setup_directories(base_dir)

    # 2. Get all images currently at the top of dataset/images/
    # Explicitly looking for them in the root of the images folder
    all_images = glob.glob(os.path.join(img_dir, "*.png"))
    
    # Extract basenames without extension to align files
    valid_pairs = []
    missing_labels = 0
    
    for img_path in all_images:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, f"{basename}.txt")
        
        if os.path.exists(lbl_path):
            valid_pairs.append((img_path, lbl_path, basename))
        else:
            print(f"WARNING: Label missing for image {basename}.png. Skipping.")
            missing_labels += 1
            
    total_valid = len(valid_pairs)
    if total_valid == 0:
        print(f"No valid image-label pairs found directly in {img_dir} to move.")
        return

    # 3. Shuffle dataset
    random.shuffle(valid_pairs)

    # 4. Split 80/20
    split_index = int(total_valid * 0.8)
    train_split = valid_pairs[:split_index]
    val_split = valid_pairs[split_index:]

    # 5. Move files
    def move_files(split_data, dest_img_dir, dest_lbl_dir):
        for img_orig, lbl_orig, basename in split_data:
            img_dest = os.path.join(dest_img_dir, f"{basename}.png")
            lbl_dest = os.path.join(dest_lbl_dir, f"{basename}.txt")
            
            # Use shutil.move to prevent duplicating massive datasets on disk
            shutil.move(img_orig, img_dest)
            shutil.move(lbl_orig, lbl_dest)

    print("Moving training files...")
    move_files(train_split, img_train, lbl_train)
    
    print("Moving validation files...")
    move_files(val_split, img_val, lbl_val)

    # 6. Logging
    print("\n--- Split Complete ---")
    print(f"Total valid images : {total_valid}")
    print(f"Training count     : {len(train_split)} (80%)")
    print(f"Validation count   : {len(val_split)} (20%)")
    if missing_labels > 0:
        print(f"Missing labels     : {missing_labels}")


if __name__ == "__main__":
    main()
