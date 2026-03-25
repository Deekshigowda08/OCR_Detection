from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import yaml


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def reset_directory_files(directory: Path, suffixes: Sequence[str]) -> None:
    ensure_dir(directory)
    for suffix in suffixes:
        for file_path in directory.glob(f"*{suffix}"):
            file_path.unlink()


def save_image(path: Path, image: np.ndarray) -> None:
    ensure_dir(path.parent)
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Failed to write image to {path}")


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def clip_box(box: Sequence[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    x1 = min(max(float(x1), 0.0), float(width - 1))
    y1 = min(max(float(y1), 0.0), float(height - 1))
    x2 = min(max(float(x2), 0.0), float(width - 1))
    y2 = min(max(float(y2), 0.0), float(height - 1))
    return [x1, y1, x2, y2]


def xyxy_to_yolo(box: Sequence[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    return cx, cy, w, h


def yolo_line(class_id: int, box: Sequence[float], width: int, height: int) -> str:
    cx, cy, w, h = xyxy_to_yolo(box, width, height)
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def polygon_to_xyxy(points: Sequence[Sequence[float]]) -> list[float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def resolve_image_size(image: np.ndarray) -> tuple[int, int]:
    height, width = image.shape[:2]
    return width, height


def build_yolo_data_config(images_dir: Path, class_names: Sequence[str], file_names: Iterable[str]) -> Path:
    file_list = sorted(file_names)
    split_index = max(1, int(len(file_list) * 0.9))
    if len(file_list) == 1:
        split_index = 1

    temp_root = Path(tempfile.mkdtemp(prefix=f"{images_dir.parent.name}_"))
    train_file = temp_root / "train.txt"
    val_file = temp_root / "val.txt"
    data_file = temp_root / "data.yaml"

    train_items = file_list[:split_index]
    val_items = file_list[split_index:] or file_list[:1]

    train_file.write_text("\n".join(str(images_dir / item) for item in train_items), encoding="utf-8")
    val_file.write_text("\n".join(str(images_dir / item) for item in val_items), encoding="utf-8")

    data = {
        "path": str(images_dir.parent),
        "train": str(train_file),
        "val": str(val_file),
        "names": {index: name for index, name in enumerate(class_names)},
    }
    data_file.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return data_file
