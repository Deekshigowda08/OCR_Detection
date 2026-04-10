from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

from serial_number_ocr.utils.config import (
    DEFAULT_EPOCHS,
    DETECTION_DATA_DIR,
    DETECTION_MODEL_PATH,
    TEXT_CLASS_NAME,
)
from serial_number_ocr.utils.io_utils import build_yolo_data_config, ensure_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRAIN_IMAGE_SIZE = 512
TRAIN_BATCH_SIZE = 8
FALLBACK_BATCH_SIZE = 4
TRAIN_WORKERS = 2
TRAIN_PROJECT = PROJECT_ROOT / "models" / "detection"
TRAIN_NAME = "run"


def train_with_fallback(model: YOLO, *, resume: bool, data: str | None = None) -> None:
    train_kwargs = {
        "imgsz": TRAIN_IMAGE_SIZE,
        "epochs": DEFAULT_EPOCHS,
        "batch": TRAIN_BATCH_SIZE,
        "workers": TRAIN_WORKERS,
        "device": 0 if torch.cuda.is_available() else "cpu",
        "save": True,
        "save_period": 1,
        "project": str(TRAIN_PROJECT),
        "name": TRAIN_NAME,
        "exist_ok": True,
    }
    if data is not None:
        train_kwargs["data"] = data
    if resume:
        train_kwargs["resume"] = True

    try:
        model.train(**train_kwargs)
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        print("CUDA OOM detected. Retrying with batch=4...")
        train_kwargs["batch"] = FALLBACK_BATCH_SIZE
        model.train(**train_kwargs)


def main() -> None:
    print("CUDA available:", torch.cuda.is_available())
    images_dir = DETECTION_DATA_DIR / "images"
    file_names = [path.name for path in images_dir.glob("*") if path.is_file()]
    if not file_names:
        raise FileNotFoundError("No detection training images found. Run scripts/convert_dataset.py first.")

    data_config = build_yolo_data_config(images_dir, [TEXT_CLASS_NAME], file_names)
    resume_path = TRAIN_PROJECT / TRAIN_NAME / "weights" / "last.pt"

    if os.path.exists(resume_path):
        print("Resuming training...")
        model = YOLO(str(resume_path))
        train_with_fallback(model, resume=True)
    else:
        print("Starting fresh training...")
        model = YOLO("yolo11n.pt")
        train_with_fallback(model, resume=False, data=str(data_config))

    trainer = model.trainer
    if trainer is None:
        raise RuntimeError("YOLO trainer was not initialized after training.")

    best_source = Path(trainer.save_dir) / "weights" / "best.pt"
    last_source = Path(trainer.save_dir) / "weights" / "last.pt"
    local_best_path = PROJECT_ROOT / "models" / "detection" / "best.pt"
    local_last_path = PROJECT_ROOT / "models" / "detection" / "last.pt"

    ensure_dir(DETECTION_MODEL_PATH.parent)
    DETECTION_MODEL_PATH.write_bytes(best_source.read_bytes())
    ensure_dir(local_best_path.parent)
    local_best_path.write_bytes(best_source.read_bytes())
    if last_source.exists():
        local_last_path.write_bytes(last_source.read_bytes())

    print(f"Saved detection model to {DETECTION_MODEL_PATH}")
    print(f"Saved detection checkpoint to {local_last_path}")
    if DETECTION_MODEL_PATH != local_best_path:
        print(f"Mirrored detection model to {local_best_path}")


if __name__ == "__main__":
    main()
