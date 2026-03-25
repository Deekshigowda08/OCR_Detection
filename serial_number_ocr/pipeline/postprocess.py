from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from pipeline.ocr import detect_digits


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1) * (by2 - by1))
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def remove_overlaps(detections: Iterable[dict], iou_threshold: float = 0.5) -> list[dict]:
    ordered = sorted(detections, key=lambda item: item["confidence"], reverse=True)
    kept: list[dict] = []
    for candidate in ordered:
        if all(compute_iou(candidate["box"], existing["box"]) < iou_threshold for existing in kept):
            kept.append(candidate)
    return kept


def sort_left_to_right(detections: Iterable[dict]) -> list[dict]:
    return sorted(detections, key=lambda item: (item["box"][0] + item["box"][2]) / 2.0)


def compute_confidence(detections: Iterable[dict]) -> float:
    items = list(detections)
    if not items:
        return 0.0
    return float(sum(item["confidence"] for item in items) / len(items))


def remap_boxes_from_rotated_180(detections: Iterable[dict], image_shape: tuple[int, int, int] | tuple[int, int]) -> list[dict]:
    height, width = image_shape[:2]
    remapped: list[dict] = []
    for item in detections:
        x1, y1, x2, y2 = item["box"]
        remapped.append(
            {
                **item,
                "box": [width - x2, height - y2, width - x1, height - y1],
            }
        )
    return remapped


def evaluate_orientation(image: np.ndarray, rotation_code: int | None = None) -> dict:
    oriented_image = cv2.rotate(image, rotation_code) if rotation_code is not None else image
    detections = detect_digits(oriented_image)
    detections = remove_overlaps(detections)
    if rotation_code == cv2.ROTATE_180:
        detections = remap_boxes_from_rotated_180(detections, oriented_image.shape)
    detections = sort_left_to_right(detections)
    text = "".join(item["digit"] for item in detections)
    confidence = compute_confidence(detections)
    return {
        "text": text,
        "confidence": confidence,
        "boxes": detections,
        "rotation_code": rotation_code,
    }


def select_best_orientation(image: np.ndarray) -> dict:
    original = evaluate_orientation(image)
    rotated = evaluate_orientation(image, cv2.ROTATE_180)
    original_score = (len(original["text"]), original["confidence"])
    rotated_score = (len(rotated["text"]), rotated["confidence"])
    return original if original_score >= rotated_score else rotated
