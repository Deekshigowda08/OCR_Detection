from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from load_dataset import load_ocr_datasets
from utils.config import DETECTION_DATA_DIR, IMAGE_SUFFIX, LABEL_SUFFIX, OCR_DATA_DIR
from utils.io_utils import (
    clip_box,
    ensure_dir,
    polygon_to_xyxy,
    reset_directory_files,
    resolve_image_size,
    save_image,
    yolo_line,
)

LIMIT = None
MAX_IMAGE_DIMENSION = 2000


@dataclass
class WordAnnotation:
    text: str
    box: list[float]
    char_boxes: list[list[float]]


def pil_to_bgr(image_obj: Any) -> np.ndarray:
    if isinstance(image_obj, np.ndarray):
        image = image_obj
    else:
        image = np.array(image_obj)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def get_image(example: dict[str, Any]) -> np.ndarray:
    for key in ("image", "img", "pixel_values"):
        if key in example and example[key] is not None:
            return pil_to_bgr(example[key])
    raise KeyError(f"Unable to find image field in keys: {list(example.keys())}")


def normalize_text_list(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [raw_value]
    return [str(item) for item in raw_value]


def box_from_any(raw_box: Any, width: int, height: int) -> list[float] | None:
    if raw_box is None:
        return None
    if isinstance(raw_box, dict):
        if {"x1", "y1", "x2", "y2"}.issubset(raw_box):
            return clip_box([raw_box["x1"], raw_box["y1"], raw_box["x2"], raw_box["y2"]], width, height)
        if {"x", "y", "width", "height"}.issubset(raw_box):
            x1 = raw_box["x"]
            y1 = raw_box["y"]
            return clip_box([x1, y1, x1 + raw_box["width"], y1 + raw_box["height"]], width, height)
        return None

    if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4 and all(not isinstance(item, (list, tuple, dict)) for item in raw_box):
        x1, y1, x2, y2 = [float(item) for item in raw_box]
        if 0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0:
            return clip_box([x1 * width, y1 * height, x2 * width, y2 * height], width, height)
        if x2 > x1 and y2 > y1:
            return clip_box([x1, y1, x2, y2], width, height)
        return clip_box([x1, y1, x1 + x2, y1 + y2], width, height)

    if isinstance(raw_box, (list, tuple)) and raw_box and isinstance(raw_box[0], (list, tuple)):
        points = [[float(point[0]), float(point[1])] for point in raw_box]
        if all(0.0 <= point[0] <= 1.0 and 0.0 <= point[1] <= 1.0 for point in points):
            scaled = [[point[0] * width, point[1] * height] for point in points]
            return clip_box(polygon_to_xyxy(scaled), width, height)
        return clip_box(polygon_to_xyxy(points), width, height)
    return None


def infer_char_boxes(word_box: list[float], text: str) -> list[list[float]]:
    if not text:
        return []
    x1, y1, x2, y2 = word_box
    char_width = (x2 - x1) / max(len(text), 1)
    boxes = []
    for index, _character in enumerate(text):
        start_x = x1 + (index * char_width)
        boxes.append([start_x, y1, start_x + char_width, y2])
    return boxes


def build_word_annotations(example: dict[str, Any], width: int, height: int) -> list[WordAnnotation]:
    words: list[WordAnnotation] = []

    json_blob = example.get("json")
    if isinstance(json_blob, dict):
        ocr_annotation = json_blob.get("ocr_annotation", {})
        texts = normalize_text_list(ocr_annotation.get("text"))
        word_boxes = ocr_annotation.get("bounding_boxes") or ocr_annotation.get("bb_word_level") or []
        char_boxes = ocr_annotation.get("bb_character_level") or []
        cursor = 0
        for index, text in enumerate(texts):
            word_box = box_from_any(word_boxes[index], width, height) if index < len(word_boxes) else None
            if word_box is None:
                continue
            next_cursor = cursor + len(text)
            chars = []
            for char_box in char_boxes[cursor:next_cursor]:
                parsed = box_from_any(char_box, width, height)
                if parsed is not None:
                    chars.append(parsed)
            cursor = next_cursor
            words.append(WordAnnotation(text=text, box=word_box, char_boxes=chars or infer_char_boxes(word_box, text)))
        if words:
            return words

    annotations = example.get("annotations") or example.get("annotation") or example.get("labels")
    if isinstance(annotations, dict):
        annotations = [annotations]

    if isinstance(annotations, list):
        for item in annotations:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or item.get("transcription") or item.get("label") or "")
            raw_box = (
                item.get("box")
                or item.get("bbox")
                or item.get("polygon")
                or item.get("points")
                or item.get("quad")
            )
            word_box = box_from_any(raw_box, width, height)
            if word_box is None:
                continue
            char_raw = item.get("char_boxes") or item.get("characters") or []
            char_boxes = [parsed for parsed in (box_from_any(box, width, height) for box in char_raw) if parsed is not None]
            words.append(WordAnnotation(text=text, box=word_box, char_boxes=char_boxes or infer_char_boxes(word_box, text)))
        if words:
            return words

    boxes_key_candidates = ("boxes", "bboxes", "bbox", "polygons", "points", "quad")
    texts_key_candidates = ("texts", "text", "transcriptions", "labels")
    box_values = None
    text_values = None
    for key in boxes_key_candidates:
        if key in example:
            box_values = example[key]
            break
    for key in texts_key_candidates:
        if key in example:
            text_values = example[key]
            break

    if isinstance(box_values, list):
        texts = normalize_text_list(text_values)
        for index, raw_box in enumerate(box_values):
            word_box = box_from_any(raw_box, width, height)
            if word_box is None:
                continue
            text = texts[index] if index < len(texts) else ""
            words.append(WordAnnotation(text=text, box=word_box, char_boxes=infer_char_boxes(word_box, text)))
    return words


def sanitize_text(text: str) -> str:
    return "".join(character for character in text if character.isdigit())


def crop_word_image(image: np.ndarray, word_box: list[float], padding_ratio: float = 0.08) -> tuple[np.ndarray, list[float]]:
    width, height = resolve_image_size(image)
    x1, y1, x2, y2 = word_box
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio
    crop_box = clip_box([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], width, height)
    cx1, cy1, cx2, cy2 = [int(round(value)) for value in crop_box]
    cropped = image[cy1:cy2, cx1:cx2]
    if cropped.size == 0:
        raise ValueError("Computed empty OCR crop")
    return cropped, crop_box


def remap_box_to_crop(box: list[float], crop_box: list[float]) -> list[float]:
    cx1, cy1, _, _ = crop_box
    x1, y1, x2, y2 = box
    return [x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1]


def prepare_output_dirs() -> None:
    for directory in (
        DETECTION_DATA_DIR / "images",
        DETECTION_DATA_DIR / "labels",
        OCR_DATA_DIR / "images",
        OCR_DATA_DIR / "labels",
    ):
        ensure_dir(directory)
    reset_directory_files(DETECTION_DATA_DIR / "images", [IMAGE_SUFFIX])
    reset_directory_files(DETECTION_DATA_DIR / "labels", [LABEL_SUFFIX])
    reset_directory_files(OCR_DATA_DIR / "images", [IMAGE_SUFFIX])
    reset_directory_files(OCR_DATA_DIR / "labels", [LABEL_SUFFIX])


def convert_detection_sample(image: np.ndarray, words: Iterable[WordAnnotation], image_name: str) -> None:
    image_path = DETECTION_DATA_DIR / "images" / f"{image_name}{IMAGE_SUFFIX}"
    label_path = DETECTION_DATA_DIR / "labels" / f"{image_name}{LABEL_SUFFIX}"
    width, height = resolve_image_size(image)
    labels = [yolo_line(0, annotation.box, width, height) for annotation in words]
    save_image(image_path, image)
    label_path.write_text("\n".join(labels), encoding="utf-8")


def convert_ocr_sample(image: np.ndarray, annotation: WordAnnotation, image_name: str) -> bool:
    digit_text = sanitize_text(annotation.text)
    if not digit_text:
        return False

    crop_image, crop_box = crop_word_image(image, annotation.box)
    crop_width, crop_height = resolve_image_size(crop_image)
    labels: list[str] = []

    digit_boxes = annotation.char_boxes or infer_char_boxes(annotation.box, annotation.text)
    digit_cursor = 0
    for char_index, character in enumerate(annotation.text):
        if not character.isdigit():
            continue
        if char_index >= len(digit_boxes):
            return False
        relative_box = remap_box_to_crop(digit_boxes[char_index], crop_box)
        relative_box = clip_box(relative_box, crop_width, crop_height)
        if relative_box[2] <= relative_box[0] or relative_box[3] <= relative_box[1]:
            continue
        labels.append(yolo_line(int(character), relative_box, crop_width, crop_height))
        digit_cursor += 1

    if not labels or digit_cursor != len(digit_text):
        return False

    image_path = OCR_DATA_DIR / "images" / f"{image_name}{IMAGE_SUFFIX}"
    label_path = OCR_DATA_DIR / "labels" / f"{image_name}{LABEL_SUFFIX}"
    save_image(image_path, crop_image)
    label_path.write_text("\n".join(labels), encoding="utf-8")
    return True


def convert_split(dataset_name: str, dataset_split: Iterable[dict[str, Any]], counters: dict[str, int]) -> None:
    count = 0
    for row_index, example in enumerate(dataset_split):
        if LIMIT is not None and row_index >= LIMIT:
            break
        count += 1
        try:
            image = get_image(example)
            width, height = resolve_image_size(image)
            if max(width, height) > MAX_IMAGE_DIMENSION:
                counters["skipped"] += 1
                continue
            words = build_word_annotations(example, width, height)
            if not words:
                continue

            detection_name = f"{dataset_name}_{counters['detection']:07d}"
            counters["detection"] += 1
            convert_detection_sample(image, words, detection_name)

            for word_index, annotation in enumerate(words):
                ocr_name = f"{dataset_name}_{row_index:07d}_{word_index:03d}"
                if convert_ocr_sample(image, annotation, ocr_name):
                    counters["ocr"] += 1
            if count % 100 == 0:
                print(f"Processed {count} samples from {dataset_name}")
        except Exception:
            counters["skipped"] += 1
            continue

    print(f"Processed {count} samples from {dataset_name}")


def main() -> None:
    prepare_output_dirs()
    synth = load_ocr_datasets()
    counters = {"detection": 0, "ocr": 0, "skipped": 0}
    print(f"Dataset size loaded: {len(synth)}")

    convert_split("synth", synth, counters)

    print(f"Detection samples: {counters['detection']}")
    print(f"OCR samples: {counters['ocr']}")
    print(f"Skipped samples: {counters['skipped']}")


if __name__ == "__main__":
    main()
