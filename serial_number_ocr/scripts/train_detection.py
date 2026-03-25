from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMAGE_SIZE,
    DETECTION_DATA_DIR,
    DETECTION_MODEL_PATH,
    TEXT_CLASS_NAME,
)
from utils.io_utils import build_yolo_data_config, ensure_dir


def main() -> None:
    images_dir = DETECTION_DATA_DIR / "images"
    file_names = [path.name for path in images_dir.glob("*") if path.is_file()]
    if not file_names:
        raise FileNotFoundError("No detection training images found. Run scripts/convert_dataset.py first.")

    data_config = build_yolo_data_config(images_dir, [TEXT_CLASS_NAME], file_names)
    model = YOLO("yolov8n.pt")
    training_root = Path(tempfile.mkdtemp(prefix="detection_train_"))
    train_results = model.train(
        data=str(data_config),
        imgsz=DEFAULT_IMAGE_SIZE,
        epochs=DEFAULT_EPOCHS,
        batch=DEFAULT_BATCH_SIZE,
        project=str(training_root),
        name="run",
        exist_ok=True,
    )

    best_source = Path(model.trainer.save_dir) / "weights" / "best.pt"
    ensure_dir(DETECTION_MODEL_PATH.parent)
    DETECTION_MODEL_PATH.write_bytes(best_source.read_bytes())
    print(f"Saved detection model to {DETECTION_MODEL_PATH}")


if __name__ == "__main__":
    main()
