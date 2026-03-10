"""
Smart E-commerce Product Classifier — Inference Engine
=======================================================
Standalone inference utility used by both the Flask API and
direct CLI prediction.  Handles model loading, preprocessing,
and top-K prediction with latency tracking.
"""

import os
import sys
import time
import json
from typing import List, Dict, Optional

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.model import ProductClassifier
from src.dataset import IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE


class ProductInference:
    """
    Encapsulates model loading and single-image inference.

    Usage::

        engine = ProductInference("models/best_model.pth")
        result = engine.predict("path/to/image.jpg", top_k=3)
    """

    REVIEW_THRESHOLD = 0.90

    def __init__(
        self,
        model_path: str,
        class_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ── Load checkpoint ─────────────────────────────────
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        self.class_names = class_names or checkpoint.get("class_names", [])
        num_classes      = checkpoint.get("num_classes", len(self.class_names))

        self.model = ProductClassifier(
            num_classes=num_classes, pretrained=False
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

        # ── Preprocessing (same as val/test) ────────────────
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        print(f"✅ Inference engine ready  |  "
              f"{num_classes} classes  |  device={self.device}")

    # ---------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        image_input,
        top_k: int = 3,
    ) -> Dict:
        """
        Run inference on a single image.

        Args:
            image_input: File path (str), PIL Image, or file-like object.
            top_k:       Number of top predictions to return.

        Returns:
            {
              "predictions": [{"category": ..., "confidence": ...}, ...],
              "requires_review": bool,
              "latency_ms": float
            }
        """
        start = time.perf_counter()

        # ── Load image ──────────────────────────────────────
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            image = Image.open(image_input).convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # ── Forward pass ────────────────────────────────────
        probs = self.model(tensor).squeeze(0)                       # (C,)
        top_probs, top_indices = probs.topk(top_k)

        predictions = [
            {
                "category":   self.class_names[idx.item()],
                "confidence": round(prob.item(), 4),
            }
            for prob, idx in zip(top_probs, top_indices)
        ]

        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        return {
            "predictions":    predictions,
            "requires_review": predictions[0]["confidence"] < self.REVIEW_THRESHOLD,
            "latency_ms":     latency_ms,
        }


# ── CLI helper ───────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/inference.py <image_path> [model_path]")
        sys.exit(1)

    img_path   = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        PROJECT_ROOT, "models", "best_model.pth"
    )

    engine = ProductInference(model_path)
    result = engine.predict(img_path)
    print(json.dumps(result, indent=2))
