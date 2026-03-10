"""
Smart E-commerce Product Classifier — Training Pipeline
========================================================
End-to-end training loop with:
  • CrossEntropyLoss + Adam optimizer
  • ReduceLROnPlateau scheduler
  • Best-model checkpointing
  • Loss / accuracy curve export
"""

import os
import sys
import json
import time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Ensure project root is importable ────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.dataset import get_dataloaders
from src.model import build_model


# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
DATA_DIR          = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR        = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH   = os.path.join(MODELS_DIR, "best_model.pth")
METRICS_PATH      = os.path.join(MODELS_DIR, "training_metrics.json")

NUM_EPOCHS        = 15
LEARNING_RATE     = 1e-4
WEIGHT_DECAY      = 1e-4
BATCH_SIZE        = 32


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Run one training epoch. Returns loss and accuracy."""
    model.train()
    running_loss    = 0.0
    correct         = 0
    total           = 0

    progress = tqdm(loader, desc="  Train", leave=False, unit="batch")
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{100. * correct / total:.1f}%")

    epoch_loss = running_loss / total
    epoch_acc  = 100.0 * correct / total
    return {"loss": epoch_loss, "accuracy": epoch_acc}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on validation set. Returns loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct      = 0
    total        = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        # Model returns softmax probs in eval → convert to logits for loss
        loss = criterion(torch.log(outputs + 1e-8), labels)

        running_loss += loss.item() * images.size(0)
        _, predicted  = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc  = 100.0 * correct / total
    return {"loss": epoch_loss, "accuracy": epoch_acc}


def plot_curves(history: Dict[str, List[float]], save_dir: str) -> None:
    """Save loss and accuracy curves as PNG images."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss ─────────────────────────────────────────────────
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   "s-", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Accuracy ─────────────────────────────────────────────
    ax2.plot(epochs, history["train_acc"], "o-", label="Train Acc")
    ax2.plot(epochs, history["val_acc"],   "s-", label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"📊 Curves saved → {path}")


def train(
    data_dir: str = DATA_DIR,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
) -> None:
    """
    Full training pipeline.

    1. Load data  →  2. Build model  →  3. Train loop
    4. Checkpoint best model  →  5. Export metrics
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Device: {device}")

    # ── Data ────────────────────────────────────────────────
    train_loader, val_loader, _, class_names = get_dataloaders(
        data_dir, batch_size=batch_size
    )
    num_classes = len(class_names)

    # ── Model ───────────────────────────────────────────────
    model     = build_model(num_classes, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.get_trainable_params(),
                           lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ── Training history ────────────────────────────────────
    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
    }
    best_val_acc = 0.0
    start_time   = time.time()

    for epoch in range(1, num_epochs + 1):
        print(f"\n{'─' * 55}")
        print(f"Epoch {epoch}/{num_epochs}  "
              f"(lr={optimizer.param_groups[0]['lr']:.2e})")
        print(f"{'─' * 55}")

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_metrics = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(f"  Train — Loss: {train_metrics['loss']:.4f}  "
              f"Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val   — Loss: {val_metrics['loss']:.4f}  "
              f"Acc: {val_metrics['accuracy']:.2f}%")

        scheduler.step(val_metrics["loss"])

        # ── Checkpoint ──────────────────────────────────────
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "val_accuracy": best_val_acc,
                "class_names":  class_names,
                "num_classes":  num_classes,
            }, BEST_MODEL_PATH)
            print(f"  💾 Best model saved (val acc {best_val_acc:.2f}%)")

    elapsed = time.time() - start_time

    # ── Export ──────────────────────────────────────────────
    metrics = {
        "best_val_accuracy":   best_val_acc,
        "total_epochs":        num_epochs,
        "training_time_sec":   round(elapsed, 1),
        "final_learning_rate": optimizer.param_groups[0]["lr"],
        "class_names":         class_names,
        "history":             history,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n📄 Metrics saved → {METRICS_PATH}")

    plot_curves(history, MODELS_DIR)

    print(f"\n🏁 Training complete in {elapsed / 60:.1f} min  |  "
          f"Best val accuracy: {best_val_acc:.2f}%")


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else DATA_DIR
    epochs    = int(sys.argv[2]) if len(sys.argv) > 2 else NUM_EPOCHS
    train(data_dir=data_path, num_epochs=epochs)
