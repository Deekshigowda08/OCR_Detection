from __future__ import annotations

import cv2
import numpy as np


def crop_region(image: np.ndarray, box: list[float], padding_ratio: float = 0.12) -> tuple[np.ndarray, list[float]]:
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding_ratio
    pad_y = (y2 - y1) * padding_ratio

    x1 = max(0, int(round(x1 - pad_x)))
    y1 = max(0, int(round(y1 - pad_y)))
    x2 = min(image_width, int(round(x2 + pad_x)))
    y2 = min(image_height, int(round(y2 + pad_y)))

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        raise ValueError("Requested crop is empty")
    return cropped, [float(x1), float(y1), float(x2), float(y2)]


def rotate_image(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    if abs(angle_degrees) < 1e-6:
        return image.copy()

    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos_term = abs(rotation_matrix[0, 0])
    sin_term = abs(rotation_matrix[0, 1])

    new_width = int((height * sin_term) + (width * cos_term))
    new_height = int((height * cos_term) + (width * sin_term))

    rotation_matrix[0, 2] += (new_width / 2.0) - center[0]
    rotation_matrix[1, 2] += (new_height / 2.0) - center[1]

    return cv2.warpAffine(
        image,
        rotation_matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def pad_to_square(image: np.ndarray, fill_value: int = 0) -> tuple[np.ndarray, dict]:
    height, width = image.shape[:2]
    size = max(height, width)
    top = (size - height) // 2
    bottom = size - height - top
    left = (size - width) // 2
    right = size - width - left
    padded = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=(fill_value, fill_value, fill_value),
    )
    return padded, {"top": top, "left": left, "size": size, "original_width": width, "original_height": height}


def crop_rotate_pad(image: np.ndarray, box: list[float], angle: float = 0.0) -> tuple[np.ndarray, dict]:
    cropped, crop_box = crop_region(image, box)
    rotated = rotate_image(cropped, angle)
    squared, padding = pad_to_square(rotated)
    metadata = {"crop_box": crop_box, "angle": angle, "padding": padding}
    return squared, metadata
