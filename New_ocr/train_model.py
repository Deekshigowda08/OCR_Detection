from ultralytics import YOLO

def main():
    # Load a lightweight pre-trained model
    model = YOLO("yolov8n.pt")
    
    # Train the model with MPS optimized arguments
    results = model.train(
        data="data.yaml",
        epochs=50,
        imgsz=512,
        batch=12,
        device="mps",
        workers=2,
        patience=15,
        
        # Color and transformation augmentations
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.05,
        scale=0.3,
        shear=0.0,
        perspective=0.0005,
        
        # Disabled augmentations not suited for strictly oriented text/serial numbers
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        mixup=0.0
    )
    
if __name__ == "__main__":
    main()
