from __future__ import annotations

import os
import sys
from pathlib import Path

from ultralytics import YOLO

from serial_number_ocr.utils.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMAGE_SIZE,
    DETECTION_DATA_DIR,
    DETECTION_MODEL_PATH,
    TEXT_CLASS_NAME,
)
from serial_number_ocr.utils.io_utils import build_yolo_data_config, ensure_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DRIVE_PROJECT = "/content/drive/MyDrive/serial_number_project"


def main() -> None:
    images_dir = DETECTION_DATA_DIR / "images"
    file_names = [path.name for path in images_dir.glob("*") if path.is_file()]
    if not file_names:
        raise FileNotFoundError("No detection training images found. Run scripts/convert_dataset.py first.")

    data_config = build_yolo_data_config(images_dir, [TEXT_CLASS_NAME], file_names)
    resume_path = PROJECT_ROOT / "models" / "detection" / "last.pt"

    if os.path.exists(resume_path):
        model = YOLO(str(resume_path))
        model.train(resume=True)
    else:
        model = YOLO("yolo11n.pt")
        model.train(
            data=str(data_config),
            imgsz=DEFAULT_IMAGE_SIZE,
            epochs=DEFAULT_EPOCHS,
            batch=DEFAULT_BATCH_SIZE,
            project=DRIVE_PROJECT,
            name="detection_training",
            exist_ok=True,
            save=True,
            save_period=1,
        )

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
    if DETECTION_MODEL_PATH != local_best_path:
        print(f"Mirrored detection model to {local_best_path}")


if __name__ == "__main__":
    main()
