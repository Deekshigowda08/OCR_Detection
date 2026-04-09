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

## Running In Google Colab

### 1. Open Colab With GPU

In Google Colab:

- open a new notebook
- go to `Runtime -> Change runtime type`
- select `GPU`

### 2. Clone The Repository

```bash
!git clone https://github.com/ShashankMk031/OCR_Detection.git
%cd OCR_Detection/serial_number_ocr
```

### 3. Install Dependencies

```bash
!pip install -r requirements.txt
!pip install ultralytics datasets opencv-python
```

### 4. Mount Google Drive

This is recommended so converted datasets and trained weights are not lost when the Colab runtime resets.

```python
from google.colab import drive
drive.mount("/content/drive")
```

### 5. Configure Data And Model Storage

The project supports environment variables for portable storage locations. In Colab, set them before running conversion or training:

```python
import os

os.environ["SERIAL_OCR_DATA_DIR"] = "/content/drive/MyDrive/serial_number_data"
os.environ["SERIAL_OCR_MODELS_DIR"] = "/content/drive/MyDrive/serial_number_models"
os.chdir("/content/MiniProj06Sem/serial_number_ocr")
```

### 6. Convert The Dataset

```bash
!python scripts/convert_dataset.py
```

This step:

- downloads the configured datasets
- converts them into YOLO detection and OCR datasets
- stores output under the configured data directory
- uses Hugging Face split slicing to avoid downloading the full dataset

### 7. Train The Detection Model

```bash
!python scripts/train_detection.py
```

This produces:

- `models/detection/best.pt`

If `SERIAL_OCR_MODELS_DIR` is set, the model is also saved under that external models directory. The training script mirrors the trained file back into:

- `serial_number_ocr/models/detection/best.pt`

### 8. Train The OCR Model

```bash
!python scripts/train_ocr.py
```

This produces:

- `models/ocr/best.pt`

If `SERIAL_OCR_MODELS_DIR` is set, the model is also saved under that external models directory. The training script mirrors the trained file back into:

- `serial_number_ocr/models/ocr/best.pt`

### 9. Run Inference In Colab

Upload a test image:

```python
from google.colab import files
uploaded = files.upload()
```

Then run:

```bash
!python pipeline/run_pipeline.py --image test.jpg
```

Or use Python directly:

```python
from pipeline.run_pipeline import run_inference

result = run_inference("test.jpg")
print(result)
```

## Colab Execution Notes

- Use a GPU runtime in Colab. Do not use TPU for this project.
- Dataset loading is limited with Hugging Face split slicing, for example `split="train[:30000]"`.
- This prevents full dataset download and reduces runtime and storage pressure in Colab.
- The ICDAR dataset has been removed from the Colab workflow due to instability and dataset mismatch issues.
- Very large images are skipped during conversion to reduce RAM crashes in notebook sessions.

## Usage

Use the main pipeline entrypoint function `pipeline.run_pipeline.run_inference`:

```python
from pipeline.run_pipeline import run_inference

result = run_inference("image.jpg")
print(result)
```

## Loading Trained Models

After training, the model weights are expected at:

- `models/detection/best.pt`
- `models/ocr/best.pt`

Exact repository paths:

- `serial_number_ocr/models/detection/best.pt`
- `serial_number_ocr/models/ocr/best.pt`

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

The `pipeline.run_pipeline.run_inference` function automatically loads the trained detection and OCR models from the `models/` directory.

## Using The Trained Models In A Backend

The backend team does not need to manually run the detector and OCR model separately unless they want low-level control. The intended integration point is:

```python
from pipeline.run_pipeline import run_inference

result = run_inference("image.jpg")
```

This function lives in:

- `pipeline/run_pipeline.py`

This returns a backend-ready dictionary:

```python
{
    "text": "...",
    "confidence": 0.98,
    "boxes": [...],
    "processing_time": 0.1234,
}
```

Recommended backend flow:

1. save the uploaded image temporarily
2. call `run_inference(image_path)`
3. return the result as JSON from the API

The backend does not need to load `best.pt` manually if it uses `run_inference()`. The pipeline loads these files automatically:

- `serial_number_ocr/models/detection/best.pt`
- `serial_number_ocr/models/ocr/best.pt`

If the backend project lives outside `serial_number_ocr/`, either:

1. keep the trained model files in those exact repository paths, or
2. set `SERIAL_OCR_MODELS_DIR` before importing the pipeline so it can locate the weights elsewhere

Minimal backend example:

```python
from pipeline.run_pipeline import run_inference

def predict_serial_number(image_path: str) -> dict:
    return run_inference(image_path)
```

Example API-style usage:

```python
from pipeline.run_pipeline import run_inference

def handle_uploaded_image(saved_image_path: str) -> dict:
    result = run_inference(saved_image_path)
    return {
        "serial_number": result["text"],
        "confidence": result["confidence"],
        "boxes": result["boxes"],
        "processing_time": result["processing_time"],
    }
```

If the backend stores trained weights outside the repository, set:

```python
import os

os.environ["SERIAL_OCR_MODELS_DIR"] = "path/to/model/storage"
```

before importing or calling the pipeline. The expected files remain:

- `detection/best.pt`
- `ocr/best.pt`

inside that configured models directory.

## Expected Result Quality

If you follow the README flow exactly, the code will:

- download the configured dataset
- convert it into YOLO training data
- train and save the detector and OCR models
- run inference through the final pipeline

You should get a correctly structured output dictionary. Actual OCR accuracy depends on:

- dataset quality
- the 2000-sample training limit used for Colab stability
- training time and Colab GPU session quality
- the similarity between test images and training data

So the output format is guaranteed by the code, but recognition accuracy still depends on the trained model quality.

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
