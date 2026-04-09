from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ultralytics import YOLO
from serial_number_ocr.utils.config import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_IMAGE_SIZE, DIGIT_CLASS_NAMES, OCR_DATA_DIR, OCR_MODEL_PATH
from serial_number_ocr.utils.io_utils import build_yolo_data_config, ensure_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))




def main() -> None:
    images_dir = OCR_DATA_DIR / "images"
    file_names = [path.name for path in images_dir.glob("*") if path.is_file()]
    if not file_names:
        raise FileNotFoundError("No OCR training images found. Run scripts/convert_dataset.py first.")

    data_config = build_yolo_data_config(images_dir, DIGIT_CLASS_NAMES, file_names)
    model = YOLO("yolo11n.pt")
    training_root = Path(tempfile.mkdtemp(prefix="ocr_train_"))
    model.train(
        data=str(data_config),
        imgsz=DEFAULT_IMAGE_SIZE,
        epochs=DEFAULT_EPOCHS,
        batch=DEFAULT_BATCH_SIZE,
        project=str(training_root),
        name="run",
        exist_ok=True,
    )

    trainer = model.trainer
    if trainer is None:
        raise RuntimeError("YOLO trainer was not initialized after training.")

    best_source = Path(trainer.save_dir) / "weights" / "best.pt"
    local_model_path = PROJECT_ROOT / "models" / "ocr" / "best.pt"
    ensure_dir(OCR_MODEL_PATH.parent)
    OCR_MODEL_PATH.write_bytes(best_source.read_bytes())
    ensure_dir(local_model_path.parent)
    local_model_path.write_bytes(best_source.read_bytes())
    print(f"Saved OCR model to {OCR_MODEL_PATH}")
    if OCR_MODEL_PATH != local_model_path:
        print(f"Mirrored OCR model to {local_model_path}")


if __name__ == "__main__":
    main()
