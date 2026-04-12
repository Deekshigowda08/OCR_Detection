"""
Fine-tune the existing YOLO OCR model on mixed real + synthetic data.
Continues from runs/detect/train3/weights/best.pt (does NOT train from scratch).
"""
from ultralytics import YOLO
import os

def main():
    # Load the existing trained model (NOT yolov8n.pt from scratch)
    MODEL_PATH = "../runs/detect/train3/weights/best.pt"
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Trained model not found at {MODEL_PATH}")
        print("Make sure runs/detect/train3/weights/best.pt exists.")
        return
    
    print(f"Loading pre-trained model from {MODEL_PATH} for fine-tuning...")
    model = YOLO(MODEL_PATH)
    
    # Fine-tune with conservative hyperparameters
    results = model.train(
        data="data.yaml",
        epochs=30,
        imgsz=512,
        batch=8,
        device="mps",
        workers=2,
        
        # Fine-tuning specific parameters
        lr0=1e-4,            # Low learning rate to avoid catastrophic forgetting
        lrf=0.01,            # Final LR = lr0 * lrf
        patience=10,         # Early stopping patience
        freeze=10,           # Freeze first 10 layers (backbone) — only train detection head
        
        # Conservative augmentations for fine-tuning
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=3.0,
        translate=0.05,
        scale=0.2,
        shear=0.0,
        perspective=0.0003,
        
        # Disabled augmentations (bad for oriented text)
        fliplr=0.0,
        flipud=0.0,
        mosaic=0.0,
        mixup=0.0,
        
        # Project naming
        project="../runs/detect",
        name="finetune",
        exist_ok=True,
        
        # Save best + last
        save=True,
        save_period=5,       # Checkpoint every 5 epochs
        plots=True,          # Auto-generate confusion matrix, PR curves, etc.
        
        # Validation
        val=True,
    )
    
    # Print final metrics
    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"Results saved to: runs/detect/finetune/")
    print(f"Best weights: runs/detect/finetune/weights/best.pt")
    print(f"Confusion matrix: runs/detect/finetune/confusion_matrix.png")
    print(f"PR curve: runs/detect/finetune/PR_curve.png")
    print(f"F1 curve: runs/detect/finetune/F1_curve.png")
    print("=" * 60)

if __name__ == "__main__":
    main()
