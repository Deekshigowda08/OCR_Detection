"""
Generate publication-quality training metric graphs from YOLO results.csv.
Reads from both original train3 and finetune runs if available.
"""
import csv
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

REPORTS_DIR = "../reports"
GRAPHS_DIR = os.path.join(REPORTS_DIR, "training_graphs")
CONFUSION_DIR = REPORTS_DIR
PREDICTIONS_DIR = os.path.join(REPORTS_DIR, "sample_predictions")

# Scan both the original training run and the fine-tune run
RUNS = [
    {"name": "Initial Training (train3)", "path": "../runs/detect/train3"},
    {"name": "Fine-Tuned", "path": "../runs/detect/finetune"},
]

def parse_results_csv(csv_path):
    """Parse YOLO results.csv into a structured dict of metric lists."""
    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        raw_header = next(reader)
        header = [h.strip() for h in raw_header]
        
        for h in header:
            data[h] = []
        
        for row in reader:
            if len(row) != len(header):
                continue
            for h, val in zip(header, row):
                try:
                    data[h].append(float(val.strip()))
                except ValueError:
                    data[h].append(0.0)
    return data

def plot_metric(epochs, values, title, ylabel, color, save_path, label=None):
    """Plot a single metric vs epochs."""
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, values, color=color, linewidth=2, label=label or ylabel)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_combined_losses(data, epochs, save_path):
    """Plot all loss curves on a single graph."""
    plt.figure(figsize=(10, 6))
    
    loss_keys = {
        "train/box_loss": ("Box Loss", "#e74c3c"),
        "train/cls_loss": ("Class Loss", "#3498db"),
        "train/dfl_loss": ("DFL Loss", "#2ecc71"),
    }
    
    for key, (label, color) in loss_keys.items():
        if key in data and data[key]:
            plt.plot(epochs[:len(data[key])], data[key], color=color, linewidth=2, label=label)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training Losses vs Epoch', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_combined_map(data, epochs, save_path):
    """Plot mAP50 and mAP50-95 together."""
    plt.figure(figsize=(10, 6))
    
    map_keys = {
        "metrics/mAP50(B)": ("mAP@0.5", "#8e44ad"),
        "metrics/mAP50-95(B)": ("mAP@0.5:0.95", "#e67e22"),
    }
    
    for key, (label, color) in map_keys.items():
        if key in data and data[key]:
            plt.plot(epochs[:len(data[key])], data[key], color=color, linewidth=2, label=label)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('mAP', fontsize=12)
    plt.title('mAP vs Epoch', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def generate_graphs_for_run(run_info):
    """Generate all training graphs for a single run."""
    run_name = run_info["name"]
    run_path = run_info["path"]
    csv_path = os.path.join(run_path, "results.csv")
    
    if not os.path.exists(csv_path):
        print(f"  ⚠️  results.csv not found at {csv_path}. Skipping {run_name}.")
        return
    
    safe_name = run_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    run_graphs_dir = os.path.join(GRAPHS_DIR, safe_name)
    os.makedirs(run_graphs_dir, exist_ok=True)
    
    print(f"\n📊 Generating graphs for: {run_name}")
    data = parse_results_csv(csv_path)
    epochs = data.get("epoch", list(range(len(next(iter(data.values()))))))
    
    # 1. Combined Loss Plot
    plot_combined_losses(data, epochs, os.path.join(run_graphs_dir, "losses_vs_epoch.png"))
    
    # 2. Individual Loss Plots
    for key, (label, color) in [
        ("train/box_loss", ("Box Loss", "#e74c3c")),
        ("train/cls_loss", ("Class Loss", "#3498db")),
        ("train/dfl_loss", ("DFL Loss", "#2ecc71")),
    ]:
        if key in data and data[key]:
            plot_metric(epochs[:len(data[key])], data[key], f"{label} vs Epoch", label, color,
                       os.path.join(run_graphs_dir, f"{key.replace('/', '_')}.png"))
    
    # 3. mAP Combined
    plot_combined_map(data, epochs, os.path.join(run_graphs_dir, "map_vs_epoch.png"))
    
    # 4. Precision + Recall
    for key, label, color in [
        ("metrics/precision(B)", "Precision", "#1abc9c"),
        ("metrics/recall(B)", "Recall", "#e74c3c"),
    ]:
        if key in data and data[key]:
            plot_metric(epochs[:len(data[key])], data[key], f"{label} vs Epoch", label, color,
                       os.path.join(run_graphs_dir, f"{label.lower()}_vs_epoch.png"))
    
    # 5. Copy confusion matrix, PR curve, F1 curve if they exist
    artifacts = ["confusion_matrix.png", "PR_curve.png", "F1_curve.png",
                 "confusion_matrix_normalized.png", "results.png"]
    for artifact in artifacts:
        src = os.path.join(run_path, artifact)
        if os.path.exists(src):
            dst = os.path.join(run_graphs_dir, artifact)
            shutil.copy2(src, dst)
            print(f"  Copied: {artifact}")
    
    # Copy confusion matrix to top-level reports/
    cm_src = os.path.join(run_path, "confusion_matrix.png")
    if os.path.exists(cm_src):
        shutil.copy2(cm_src, os.path.join(CONFUSION_DIR, f"confusion_matrix_{safe_name}.png"))

def copy_sample_predictions(run_path):
    """Copy sample prediction images from validation runs."""
    val_dir = os.path.join(run_path, "val_batch0_pred.jpg")
    if os.path.exists(val_dir):
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        shutil.copy2(val_dir, os.path.join(PREDICTIONS_DIR, "val_predictions.jpg"))
        print(f"  Copied validation predictions to {PREDICTIONS_DIR}")
    
    # Also check for individual prediction images
    pred_dir = os.path.join(run_path, "predictions")
    if os.path.exists(pred_dir):
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        for f in os.listdir(pred_dir)[:10]:  # Max 10 samples
            shutil.copy2(os.path.join(pred_dir, f), os.path.join(PREDICTIONS_DIR, f))

def main():
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("TRAINING GRAPH GENERATION")
    print("=" * 60)
    
    for run in RUNS:
        generate_graphs_for_run(run)
        copy_sample_predictions(run["path"])
    
    print(f"\n{'='*60}")
    print("REPORT GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Training graphs: {os.path.abspath(GRAPHS_DIR)}/")
    print(f"Confusion matrix: {os.path.abspath(CONFUSION_DIR)}/")
    print(f"Sample predictions: {os.path.abspath(PREDICTIONS_DIR)}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
