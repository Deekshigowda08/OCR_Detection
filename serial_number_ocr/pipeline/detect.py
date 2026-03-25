from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from utils.config import DEFAULT_CONFIDENCE, DETECTION_MODEL_PATH


@lru_cache(maxsize=1)
def load_detection_model(model_path: str | Path = DETECTION_MODEL_PATH) -> YOLO:
    return YOLO(str(model_path))


def detect_text_regions(image: np.ndarray, model_path: str | Path = DETECTION_MODEL_PATH, conf: float = DEFAULT_CONFIDENCE) -> list[dict]:
    model = load_detection_model(model_path)
    results = model.predict(source=image, conf=conf, verbose=False)
    detections: list[dict] = []

    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            detections.append(
                {
                    "box": [float(value) for value in xyxy],
                    "confidence": float(box.conf[0].item()),
                    "class_id": int(box.cls[0].item()),
                    "angle": 0.0,
                }
            )
    return detections
