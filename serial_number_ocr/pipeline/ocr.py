from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from utils.config import DEFAULT_CONFIDENCE, OCR_MODEL_PATH


@lru_cache(maxsize=1)
def load_ocr_model(model_path: str | Path = OCR_MODEL_PATH) -> YOLO:
    return YOLO(str(model_path))


def detect_digits(image: np.ndarray, model_path: str | Path = OCR_MODEL_PATH, conf: float = DEFAULT_CONFIDENCE) -> list[dict]:
    model = load_ocr_model(model_path)
    results = model.predict(source=image, conf=conf, verbose=False)
    detections: list[dict] = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "digit": str(int(box.cls[0].item())),
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(box.conf[0].item()),
                }
            )
    return detections
