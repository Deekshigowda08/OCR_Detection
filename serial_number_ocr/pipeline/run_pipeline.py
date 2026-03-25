from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.crop_rotate import crop_rotate_pad
from pipeline.detect import detect_text_regions
from pipeline.postprocess import select_best_orientation
from utils.io_utils import load_image


def remap_boxes_to_image(detections: list[dict], crop_metadata: dict) -> list[dict]:
    crop_x1, crop_y1, _, _ = crop_metadata["crop_box"]
    left_padding = crop_metadata["padding"]["left"]
    top_padding = crop_metadata["padding"]["top"]
    remapped: list[dict] = []

    for item in detections:
        x1, y1, x2, y2 = item["box"]
        remapped.append(
            {
                **item,
                "box": [
                    round(float(x1 - left_padding + crop_x1), 2),
                    round(float(y1 - top_padding + crop_y1), 2),
                    round(float(x2 - left_padding + crop_x1), 2),
                    round(float(y2 - top_padding + crop_y1), 2),
                ],
            }
        )
    return remapped


def run_inference(image_path: str) -> dict:
    started_at = time.perf_counter()
    image = load_image(Path(image_path))
    text_regions = detect_text_regions(image)

    best_result = {"text": "", "confidence": 0.0, "boxes": []}
    for region in text_regions:
        crop, crop_metadata = crop_rotate_pad(image, region["box"], region.get("angle", 0.0))
        candidate = select_best_orientation(crop)
        candidate["boxes"] = remap_boxes_to_image(candidate["boxes"], crop_metadata)
        if (len(candidate["text"]), candidate["confidence"]) > (len(best_result["text"]), best_result["confidence"]):
            best_result = candidate

    return {
        "text": best_result["text"],
        "confidence": round(float(best_result["confidence"]), 4),
        "boxes": best_result["boxes"],
        "processing_time": round(time.perf_counter() - started_at, 4),
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python pipeline/run_pipeline.py <image_path>")
    result = run_inference(sys.argv[1])
    print(result)


if __name__ == "__main__":
    main()
