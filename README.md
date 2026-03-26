# Automated Serial Number Recognition System

This repository contains the machine learning pipeline for automated serial number recognition from industrial metal surfaces. The current implementation lives in `serial_number_ocr/` and covers dataset conversion, YOLOv12 training, and inference for detecting and reading engraved numeric serial numbers from images.

The repository is structured to support a larger product build. The ML pipeline is already isolated as its own module, and separate `backend/` and `frontend/` folders can be added at the repository root by teammates without changing the ML project layout.

## Repository Structure

```text
MiniProj06Sem/
├── serial_number_ocr/        # ML pipeline for dataset prep, training, and inference
│   ├── data/
│   ├── models/
│   ├── pipeline/
│   ├── scripts/
│   ├── utils/
│   ├── load_dataset.py
│   ├── README.md
│   └── requirements.txt
├── backend/                  # To be added by backend team
├── frontend/                 # To be added by frontend team
└── README.md
```

## ML Pipeline Overview

The `serial_number_ocr/` project implements an end-to-end OCR workflow:

- converts source datasets into YOLO training format
- trains a text-region detector
- trains a digit OCR model
- runs inference and returns structured JSON output

The inference pipeline is designed for engraved serial numbers and includes crop handling, dual-orientation OCR, overlap filtering, and confidence scoring.

## Current Scope

Implemented now:

- dataset conversion scripts
- YOLOv11 training entrypoints
- modular OCR inference pipeline
- API-ready inference output

Planned as separate root modules:

- `backend/` for API and service integration
- `frontend/` for the user-facing application

## Setup For The ML Module

Install dependencies from inside `serial_number_ocr/`:

```bash
cd serial_number_ocr
pip install -r requirements.txt
```

## Training Workflow

Run the following from `serial_number_ocr/`:

```bash
python scripts/convert_dataset.py
python scripts/train_detection.py
python scripts/train_ocr.py
```

## Inference Usage

Example:

```python
from pipeline.run_pipeline import run_inference

result = run_inference("image.jpg")
print(result)
```

Expected output shape:

```python
{
    "text": "...",
    "confidence": 0.98,
    "boxes": [...],
    "processing_time": 0.1234,
}
```

## Notes

- Trained model files are not committed to the repository.
- Datasets are not stored in the repository.
- The ML code is written to stay portable across local development, Colab training, and later backend integration.
- See [serial_number_ocr/README.md](/Users/shashankmk/Documents/Projects-Development/MiniProj06Sem/serial_number_ocr/README.md) for ML-project-specific details.
