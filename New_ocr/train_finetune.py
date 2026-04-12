"""
Fine-tune the existing YOLO OCR model.
Cross-platform: auto-detects CUDA (Windows/Linux) or MPS (macOS).
Continues from runs/detect/train3/weights/best.pt — does NOT train from scratch.
"""
import os
import torch
from ultralytics import YOLO

def main():
    # Resolve model path (cross-platform)
    MODEL_PATH = os.path.join("..", "runs", "detect", "train3", "weights", "best.pt")
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Trained model not found at {MODEL_PATH}")
        print("Make sure runs/detect/train3/weights/best.pt exists.")
        return
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = "0"  # NVIDIA GPU (Windows/Linux)
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon
        print("Using Apple MPS")
    else:
        device = "cpu"
        print("WARNING: No GPU detected. Training on CPU (very slow).")
    
    print(f"\nLoading pre-trained model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # Fine-tune with conservative hyperparameters
    # batch=8 fits within RTX 3050 6GB VRAM at imgsz=512
    results = model.train(
        data="data.yaml",
        epochs=25,
        imgsz=512,
        batch=8,
        device=device,
        workers=4 if os.name == "nt" else 2,  # More workers on Windows
        
        # Fine-tuning specific parameters
        lr0=1e-4,            # Low LR to avoid catastrophic forgetting
        lrf=0.01,            # Final LR = lr0 * lrf
        patience=10,         # Early stopping patience
        freeze=10,           # Freeze first 10 layers (backbone)
        
        # Conservative augmentations
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
        
        # Output configuration
        project=os.path.join("..", "runs", "detect"),
        name="finetune",
        exist_ok=True,
        save=True,
        save_period=5,
        plots=True,
        val=True,
    )
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    
    finetune_dir = os.path.join("..", "runs", "detect", "finetune")
    print(f"Results:          {os.path.abspath(finetune_dir)}")
    print(f"Best weights:     {os.path.join(finetune_dir, 'weights', 'best.pt')}")
    print(f"Confusion matrix: {os.path.join(finetune_dir, 'confusion_matrix.png')}")
    print(f"PR curve:         {os.path.join(finetune_dir, 'PR_curve.png')}")
    print(f"F1 curve:         {os.path.join(finetune_dir, 'F1_curve.png')}")
    print(f"Results plot:     {os.path.join(finetune_dir, 'results.png')}")
    print("=" * 60)
    
    # OOM Recovery tip
    print("\nIf you got CUDA Out of Memory:")
    print("  Reduce batch: batch=6 or batch=4")
    print("  Reduce imgsz: imgsz=416")

if __name__ == "__main__":
    main()
