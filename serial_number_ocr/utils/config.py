from __future__ import annotations

from pathlib import Path


# All project paths are resolved relative to this file so the code can run
# from the repository on different operating systems without hardcoded paths.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DETECTION_DATA_DIR = DATA_DIR / "detection"
OCR_DATA_DIR = DATA_DIR / "ocr"
MODELS_DIR = PROJECT_ROOT / "models"
# Trained weights are expected to be copied here by the training scripts.
DETECTION_MODEL_PATH = MODELS_DIR / "detection" / "best.pt"
OCR_MODEL_PATH = MODELS_DIR / "ocr" / "best.pt"

TEXT_CLASS_NAME = "TEXT"
DIGIT_CLASS_NAMES = [str(index) for index in range(10)]

IMAGE_SUFFIX = ".jpg"
LABEL_SUFFIX = ".txt"

DEFAULT_IMAGE_SIZE = 640
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 50
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.5
