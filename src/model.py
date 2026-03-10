"""
Smart E-commerce Product Classifier — Model Architecture
=========================================================
ResNet50 transfer-learning model with frozen early layers
and a custom classification head for e-commerce categories.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


class ProductClassifier(nn.Module):
    """
    ResNet50-based classifier for e-commerce product images.

    Architecture decisions
    ─────────────────────
    • **Feature extractor** — Layers up to and including ``layer3`` are
      frozen so training only updates ``layer4`` and the custom head.
    • **Classification head** — ``fc`` is replaced with::

          Linear(2048 → 512) → ReLU → Dropout(0.3) → Linear(512 → num_classes)

    • At inference time ``forward()`` returns **softmax probabilities**.
      During training raw logits are returned (``CrossEntropyLoss``
      applies its own log-softmax).
    """

    def __init__(self, num_classes: int, pretrained: bool = True,
                 dropout: float = 0.3):
        super().__init__()

        # ── Load pretrained backbone ────────────────────────
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # ── Freeze layers through layer3 ────────────────────
        frozen_layers = [
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
        ]
        for layer in frozen_layers:
            for param in layer.parameters():
                param.requires_grad = False

        # ── Custom classification head ──────────────────────
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    # ---------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            Softmax probabilities during eval, raw logits during training.
        """
        logits = self.backbone(x)
        if not self.training:
            return torch.softmax(logits, dim=1)
        return logits

    # ---------------------------------------------------------
    def get_trainable_params(self):
        """Return only parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self) -> dict:
        """Return counts of total vs. trainable parameters."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_params":     total,
            "trainable_params": trainable,
            "frozen_params":    total - trainable,
            "trainable_pct":    f"{100 * trainable / total:.1f}%",
        }


def build_model(
    num_classes: int,
    pretrained: bool = True,
    device: Optional[torch.device] = None,
) -> ProductClassifier:
    """
    Factory helper — builds the model and moves it to *device*.

    Args:
        num_classes: Number of product categories.
        pretrained:  Whether to load ImageNet weights.
        device:      Target device (auto-detected when ``None``).

    Returns:
        A ``ProductClassifier`` on the specified device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ProductClassifier(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)

    stats = model.count_parameters()
    print(f"✅ Model built — {stats['trainable_params']:,} trainable "
          f"/ {stats['total_params']:,} total params "
          f"({stats['trainable_pct']})")

    return model


# ── Quick sanity check ───────────────────────────────────────
if __name__ == "__main__":
    model = build_model(num_classes=10)
    dummy = torch.randn(2, 3, 224, 224).to(
        next(model.parameters()).device
    )
    out = model(dummy)
    print(f"   Output shape: {out.shape}")        # (2, 10)
    print(f"   Sum of probs: {out.sum(dim=1)}")    # ≈ [1, 1]
