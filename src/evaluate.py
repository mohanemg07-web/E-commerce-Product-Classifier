"""
Smart E-commerce Product Classifier — Model Evaluation
=======================================================
Generates comprehensive test-set metrics:
  • Accuracy, Precision, Recall, F1 (macro & per-class)
  • Confusion matrix heatmap
  • Misclassified-image gallery for debugging
  • JSON metric export
"""

import os
import sys
import json
from typing import List

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from PIL import Image
from torchvision import transforms

# ── Project imports ──────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.dataset import get_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from src.model import ProductClassifier


# ── Paths ────────────────────────────────────────────────────
MODELS_DIR       = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH  = os.path.join(MODELS_DIR, "best_model.pth")
METRICS_PATH     = os.path.join(MODELS_DIR, "training_metrics.json")
MISCLASS_DIR     = os.path.join(MODELS_DIR, "misclassified")


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized image tensor back to [0, 255] uint8 numpy array."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = tensor.cpu() * std + mean
    img  = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def load_model(model_path: str, device: torch.device):
    """Load the best checkpoint."""
    checkpoint   = torch.load(model_path, map_location=device, weights_only=False)
    num_classes  = checkpoint["num_classes"]
    class_names  = checkpoint["class_names"]

    model = ProductClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded from {model_path}  "
          f"(epoch {checkpoint['epoch']}, "
          f"val acc {checkpoint['val_accuracy']:.2f}%)")
    return model, class_names


@torch.no_grad()
def evaluate(
    data_dir: str = None,
    model_path: str = BEST_MODEL_PATH,
) -> dict:
    """
    Run full evaluation on the test split.

    Returns:
        Dictionary of computed metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if data_dir is None:
        data_dir = os.path.join(PROJECT_ROOT, "data")

    # ── Load data & model ───────────────────────────────────
    _, _, test_loader, class_names_ds = get_dataloaders(data_dir)
    model, class_names = load_model(model_path, device)

    all_preds:  List[int] = []
    all_labels: List[int] = []
    misclassified = []    # store up to 20 examples

    for images, labels in test_loader:
        images = images.to(device)
        probs  = model(images)                       # softmax probs
        _, predicted = probs.max(1)

        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.tolist())

        # Collect misclassified samples
        if len(misclassified) < 20:
            wrong = (predicted.cpu() != labels)
            for idx in wrong.nonzero(as_tuple=False).squeeze(1).tolist():
                if len(misclassified) >= 20:
                    break
                misclassified.append({
                    "image":     images[idx].cpu(),
                    "true":      labels[idx].item(),
                    "predicted": predicted[idx].cpu().item(),
                    "confidence": probs[idx].max().cpu().item(),
                })

    # ── Metrics ─────────────────────────────────────────────
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    precision_per, recall_per, f1_per, support_per = \
        precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )

    report = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    )

    print(f"\n{'═' * 55}")
    print(f"  TEST RESULTS")
    print(f"{'═' * 55}")
    print(f"  Accuracy : {accuracy:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"{'─' * 55}")
    print(report)

    # ── Confusion matrix ────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    _plot_confusion_matrix(cm, class_names)

    # ── Save misclassified images ───────────────────────────
    _save_misclassified(misclassified, class_names)

    # ── Export JSON ──────────────────────────────────────────
    metrics = {
        "test_accuracy":  round(accuracy, 2),
        "macro_precision": round(float(precision), 4),
        "macro_recall":    round(float(recall), 4),
        "macro_f1":        round(float(f1), 4),
        "per_class": {
            name: {
                "precision": round(float(precision_per[i]), 4),
                "recall":    round(float(recall_per[i]), 4),
                "f1":        round(float(f1_per[i]), 4),
                "support":   int(support_per[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
    }

    # Merge with existing training metrics if present
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            existing = json.load(f)
        existing["evaluation"] = metrics
        full_metrics = existing
    else:
        full_metrics = {"evaluation": metrics}

    with open(METRICS_PATH, "w") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"\n📄 Metrics exported → {METRICS_PATH}")

    return metrics


def _plot_confusion_matrix(cm: np.ndarray, class_names: list) -> None:
    """Generate and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Confusion matrix saved → {path}")


def _save_misclassified(samples: list, class_names: list) -> None:
    """Save a grid and individual images of misclassified examples."""
    if not samples:
        print("🎉 No misclassified samples found!")
        return

    os.makedirs(MISCLASS_DIR, exist_ok=True)

    # Save individual images
    for i, s in enumerate(samples[:10]):
        img_np = _denormalize(s["image"])
        img    = Image.fromarray(img_np)
        fname  = (f"{i:02d}_true-{class_names[s['true']]}"
                  f"_pred-{class_names[s['predicted']]}.png")
        img.save(os.path.join(MISCLASS_DIR, fname))

    # Save grid
    n = min(len(samples), 10)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = np.array(axes).flatten()

    for i in range(len(axes)):
        axes[i].axis("off")
        if i < n:
            img_np = _denormalize(samples[i]["image"])
            axes[i].imshow(img_np)
            axes[i].set_title(
                f"True: {class_names[samples[i]['true']]}\n"
                f"Pred: {class_names[samples[i]['predicted']]}\n"
                f"Conf: {samples[i]['confidence']:.2f}",
                fontsize=8,
            )

    plt.suptitle("Misclassified Examples", fontsize=14)
    plt.tight_layout()
    path = os.path.join(MISCLASS_DIR, "misclassified_grid.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"🖼  Misclassified examples saved → {MISCLASS_DIR}/")


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    evaluate(data_dir=data_path)
