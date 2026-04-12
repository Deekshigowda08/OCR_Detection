# Fine-Tuning Pipeline — Windows Setup Guide

Complete step-by-step guide for fine-tuning the YOLO OCR model on a **Windows** machine with **NVIDIA RTX 3050 (6GB VRAM)**.

---

## System Requirements

| Component | Requirement                |
| --------- | -------------------------- |
| OS        | Windows 10/11              |
| Python    | 3.10                       |
| GPU       | NVIDIA RTX 3050 (6GB VRAM) |
| CUDA      | 12.1 (via PyTorch)         |
| Disk      | ~5GB free                  |

---

## Step 1: Environment Setup

Open **PowerShell** or **Command Prompt**.

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install ultralytics opencv-python matplotlib numpy pillow tqdm pyyaml

# Verify GPU is detected
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

**Expected output:**

```
CUDA: True
GPU: NVIDIA GeForce RTX 3050
```

> If `CUDA: False`, reinstall PyTorch matching your CUDA version: https://pytorch.org/get-started/locally/

---

## Step 2: Clone Repository

```bash
git clone https://github.com/ShashankMk031/OCR_Detection.git
cd OCR_Detection
```

---

## Step 3: Project Structure

Ensure the following structure exists:

```
OCR_Detection/
├── New_ocr/
│   ├── auto_label.py
│   ├── prepare_dataset.py
│   ├── train_finetune.py
│   ├── data.yaml
│   └── dataset/
│       ├── real_images/          ← PUT DRIVE IMAGES HERE
│       ├── real_labels/          ← auto-generated
│       ├── real_images_processed/← auto-generated
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
└── runs/
    └── detect/
        └── train3/
            └── weights/
                └── best.pt       ← existing trained model
```

---

## Step 4: Add Real-World Images

1. Download images from the shared Google Drive folder
2. Copy ALL images into:
   ```
   New_ocr/dataset/real_images/
   ```
3. Supported formats: `.png`, `.jpg`, `.jpeg`, `.webp`

> **IMPORTANT:** Ensure there are NO SPACES in folder paths.

---

## Step 5: Auto-Label Images

```bash
cd New_ocr
python auto_label.py
```

**What this does:**

- Loads `runs/detect/train3/weights/best.pt`
- Runs inference on each image in `dataset/real_images/`
- Generates YOLO `.txt` label files in `dataset/real_labels/`
- Saves preprocessed images in `dataset/real_images_processed/`
- Filters detections with confidence ≥ 0.5
- Removes duplicate overlapping boxes

**Output:**

```
✅ Total images successfully labeled: XX
❌ Skipped (No text / Empty): XX
```

> **After this step:** Open the generated `.txt` files in [LabelImg](https://github.com/HumanSignal/labelImg) or [Roboflow](https://roboflow.com) to manually fix incorrect boxes before training.

---

## Step 6: Prepare Dataset

```bash
python prepare_dataset.py
```

**What this does:**

- Merges real-world labeled images with existing synthetic data
- Shuffles the combined dataset
- Splits 80% train / 20% val
- Copies files into `dataset/images/train`, `dataset/images/val`, etc.

**Output:**

```
DATASET PREPARATION COMPLETE
Final train images: XXXX
Final val images:   XXX
```

---

## Step 7: Fine-Tune the Model

```bash
python train_finetune.py
```

**Training parameters:**

| Parameter | Value | Reason                       |
| --------- | ----- | ---------------------------- |
| epochs    | 25    | Sufficient for fine-tuning   |
| batch     | 8     | Fits 6GB VRAM at imgsz=512   |
| imgsz     | 512   | Matches original training    |
| lr0       | 1e-4  | Low LR prevents forgetting   |
| freeze    | 10    | Locks backbone, trains head  |
| patience  | 10    | Early stop if no improvement |
| device    | 0     | Auto-detected NVIDIA GPU     |

**Outputs saved to `runs/detect/finetune/`:**

- `weights/best.pt` — best model weights
- `weights/last.pt` — last epoch weights
- `results.png` — all metric curves
- `confusion_matrix.png` — class confusion matrix
- `PR_curve.png` — Precision-Recall curve
- `F1_curve.png` — F1 score curve

> **If you get CUDA Out of Memory:**
> Edit `train_finetune.py` and reduce `batch=6` or `batch=4`.

---

## Step 8: Verify Results

Check the generated files:

```bash
dir ..\runs\detect\finetune\
dir ..\runs\detect\finetune\weights\
```

Confirm these exist:

- [x] `best.pt`
- [x] `results.png`
- [x] `confusion_matrix.png`

---

## Step 9: Push to GitHub

```bash
cd ..
git add .
git commit -m "Fine-tuned model on real-world data"
git push origin main
```

> **Note:** If `best.pt` is too large for GitHub (>100MB), add it to `.gitignore` and use [Git LFS](https://git-lfs.com) or share via Google Drive.

---

## Quick Reference — 3 Commands

```bash
cd New_ocr

# 1. Auto-label images from drive
python auto_label.py

# 2. Prepare train/val split
python prepare_dataset.py

# 3. Fine-tune model
python train_finetune.py
```

---

## Troubleshooting

| Issue                 | Fix                                                                                       |
| --------------------- | ----------------------------------------------------------------------------------------- |
| `CUDA: False`         | Reinstall PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| CUDA Out of Memory    | Reduce `batch` to 6 or 4 in `train_finetune.py`                                           |
| `ModuleNotFoundError` | Run `pip install ultralytics opencv-python matplotlib`                                    |
| Model not found       | Ensure `runs/detect/train3/weights/best.pt` exists                                        |
| No images found       | Place images in `New_ocr/dataset/real_images/`                                            |
| Training stuck        | Check GPU temp; reduce `workers` to 2                                                     |
| Spaces in path        | Move project to a path without spaces (e.g., `C:\Projects\`)                              |
