from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from inference import run_inference
from utils import decode_image, generate_metrics_graphs

app = FastAPI(title="YOLO OCR Serial API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    image_bytes = await image.read()
    img_bgr = decode_image(image_bytes)
    
    if img_bgr is None:
        return {"error": "Invalid image format uploaded"}
        
    result = run_inference(img_bgr)
    return result

@app.get("/metrics")
async def metrics():
    graphs = generate_metrics_graphs()
    if not graphs:
        return {"error": "Metrics dataset missing or model not processed"}
    return graphs
