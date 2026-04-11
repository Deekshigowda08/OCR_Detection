# JustOCR: Industrial Serial Number Detection

An end-to-end, production-ready Optical Character Recognition (OCR) pipeline built from scratch to precisely detect strictly alphanumeric serial numbers layered on heavily degraded metallic surfaces.

The system encapsulates an advanced Digital Image Processing (DIP) pipeline merged directly into a custom-trained Ultralytics YOLOv8 architecture, served outwardly via a lightning-fast FastAPI backend and a Next.js front-end client interface.

---

## Project Architecture

The architecture is naturally compartmentalized into three primary execution modules:

1. **Synthetic Data Generation (`New_ocr/`)**:
   Driven by `gen_data.py`, this engine generates perfectly bounded YOLO classification images. It heavily randomizes alphanumeric data strings, strictly handles rotational intersection mathematics for bounding boxes, and artificially synthesizes rotational noise overlays.
2. **Backend Inference Engine (`backend/`)**:
   A FastAPI application (`main.py` + `inference.py`) that strictly loads the trained YOLO target weights into active memory to bypass disk latency overhead. Features advanced algorithmic overlap filtering (Intersection-over-Union mapping) to cull duplicate boxes mathematically, and an auto-recovery mechanism protecting against natively upside-down 180° images.
3. **Next.js Web Client (`frontend/`)**:
   A fully responsive React client interface handling multipart file uploads, fetching base64-encoded visual arrays from the REST API, and rendering the native Model Evaluation/Confidence matrices perfectly on-screen.

---

## Advanced Preprocessing Pipeline

Real-world optical character recognition heavily relies on preparing the data. Identifying strings resting on metallic vessels (rusted steel, engraved dog tags, industrial parts) requires actively stripping away deep specular highlights and high-amplitude scratches that usually confuse generic models.

We process every loaded frame linearly (`backend/utils.py` and `New_ocr/preprocess.py`) via the following array:

1. **Bilateral Filtering**: Mathematically erases surface rust and broad background textures while keeping core character edge delineations razor sharp.
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: Sweeps across the frame grid-by-grid to rip out extreme localized light glare and deep shadows ubiquitous to curved metallic hardware.
3. **Adaptive Gaussian Thresholding (21x21 Block)**: Bypasses naive global binarization (Otsu) with a localized Gaussian search algorithm, enforcing isolated character strokes regardless of immediate glare constraints.
4. **Morphological Median Blurring**: Precisely targets and extracts the last remaining "salt-and-pepper" visual artifacts inherently induced by heavy adaptive thresholding.

---

## The YOLO Concept Model

The localized inference capabilities run purely through **Ultralytics YOLOv8 Nano (`yolov8n.pt`)**.
The footprint of the Nano iteration was chosen completely around its speed profiles matching to mobile/standard CPU architectures. The network isolates its predictions to exactly **36 definitive feature classes**:

- **`0-9`** (Standard Extracted Numeric Digits)
- **`A-Z`** (Extracted Uppercase Alphabetic Characters)

---

## Dataset Handling & Split Mechanics

The base generation loop creates 10,000 distinct images directly into a flat directory structure.
The custom `split_dataset.py` engine cleanly compartmentalizes the data dynamically prior to training to stop over-fitting mathematically:

- **Training Set (80%)**: 8,000 images packed sequentially into `dataset/images/train/`.
- **Validation Set (20%)**: 2,000 holdout validation samples stored in `dataset/images/val/`. This strictly verifies the network generates loss mappings explicitly against data it structurally has never encountered.

---

## Model Training on Mac MPS

The network was natively compiled, stressed, and meticulously configured via script constraints explicitly to launch directly on Apple Silicon (`MPS`) architecture to avoid OOM crashes common with PyTorch accumulations on Mac.

Core Configurations (`train_model.py`):

- **Epochs**: 50 total passes.
- **Resolution**: Sized heavily to `imgsz=640` ensuring visual integrity against minor background artifacts.
- **Batch Size**: 8 (Maximizing 16GB architecture without overflowing physical buffers into hard swap memory).
- **Augmentation Overrides**: Complex alterations like `mosaic=0.0` and `mixup=0.0` were manually stripped and disabled to forcibly prevent character destruction, while localized HSV manipulation (`hsv_s=0.7`) actively pushed the model to rely intensely on shape extraction rather than artificial lighting assumptions.
