# Automated Serial Number Recognition from Metal Surfaces

This project implements an end-to-end OCR pipeline for engraved serial numbers on industrial metal surfaces. It detects text regions, recognizes digits using YOLO-based OCR, and returns structured JSON output that is ready to integrate into a backend API.

The pipeline is designed for noisy real-world images and includes region cropping, rotation handling, overlap filtering, and confidence scoring. It is portable across development and deployment environments and avoids OS-specific path handling.

## Project Structure

```text
serial_number_ocr/
├── data/
│   ├── detection/
│   │   ├── images/
│   │   └── labels/
│   └── ocr/
│       ├── images/
│       └── labels/
├── models/
│   ├── detection/
│   ├── ocr/
│   └── README.md
├── pipeline/
│   ├── crop_rotate.py
│   ├── detect.py
│   ├── ocr.py
│   ├── postprocess.py
│   └── run_pipeline.py
├── scripts/
│   ├── convert_dataset.py
│   ├── train_detection.py
│   └── train_ocr.py
├── utils/
│   ├── config.py
│   └── io_utils.py
├── load_dataset.py
├── README.md
└── requirements.txt
```

## Setup

Install dependencies from the project root:

```bash
pip install -r requirements.txt
```

## Training In Google Colab

Run the training workflow in this order from the `serial_number_ocr/` directory:

```bash
python scripts/convert_dataset.py
python scripts/train_detection.py
python scripts/train_ocr.py
```

`convert_dataset.py` downloads and converts the configured datasets into YOLO training format. The detection and OCR training scripts then train YOLOv8 models and save the best weights under `models/`.

## Usage

Example:

```python
from pipeline.run_pipeline import run_inference

result = run_inference("image.jpg")
print(result)
```

## Loading Trained Models

After training, the model weights are expected at:

- `models/detection/best.pt`
- `models/ocr/best.pt`

You can load them directly with Ultralytics:

```python
from ultralytics import YOLO

detection_model = YOLO("models/detection/best.pt")
ocr_model = YOLO("models/ocr/best.pt")
```

If you want to use the full OCR pipeline, place both trained files in the paths above and call:

```python
from pipeline.run_pipeline import run_inference

result = run_inference("image.jpg")
print(result["text"])
```

The pipeline automatically loads the trained detection and OCR models from the `models/` directory.

## Output Format

```python
{
    "text": "...",
    "confidence": 0.98,
    "boxes": [...],
    "processing_time": 0.1234,
}
```

## Notes

- Trained model weights are not included in the repository because they are too large for source control.
- Datasets are not included in the repository and are expected to be downloaded during dataset conversion.
- The inference API is designed to plug into a backend service in the repository root workspace, `MiniProj06Sem/`.
