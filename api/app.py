"""
Smart E-commerce Product Classifier — Flask REST API
=====================================================
Endpoints
─────────
  GET  /          → Web UI (drag-and-drop classifier)
  GET  /health    → API & model status
  POST /predict   → Top-3 category predictions for an uploaded image
"""

import os
import sys
import time
import json
import logging
from io import BytesIO

from flask import Flask, request, jsonify, render_template
from PIL import Image

# ── Ensure project root is importable ────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.inference import ProductInference

# ─────────────────────────────────────────────────────────────
# App Configuration
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Model paths (configurable via env vars) ──────────────────
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(PROJECT_ROOT, "models", "best_model.pth"),
)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}

# ── Load inference engine at startup ─────────────────────────
engine = None


def _load_engine():
    """Lazily initialise the inference engine."""
    global engine
    if engine is None:
        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model file not found at {MODEL_PATH}. "
                           f"/predict will return 503 until a model is available.")
            return
        logger.info(f"Loading model from {MODEL_PATH} …")
        engine = ProductInference(MODEL_PATH)
        logger.info("Inference engine ready ✅")


# Load on import so Gunicorn workers have the model ready
_load_engine()


def _allowed_file(filename: str) -> bool:
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Serve the web UI."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check — returns API and model readiness."""
    return jsonify({
        "status":       "healthy",
        "model_loaded": engine is not None,
        "model_path":   MODEL_PATH,
        "version":      "1.0.0",
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept an image and return Top-3 predictions.

    Supports two input modes:
      1. ``multipart/form-data`` with a file field named ``image``.
      2. Raw image bytes in the request body (``Content-Type: image/*``).
    """
    if engine is None:
        _load_engine()
        if engine is None:
            return jsonify({
                "error": "Model not loaded. Ensure a trained model exists "
                         "at the configured MODEL_PATH.",
            }), 503

    # ── Extract image ───────────────────────────────────────
    try:
        if "image" in request.files:
            file = request.files["image"]
            if file.filename == "":
                return jsonify({"error": "No file selected."}), 400
            if not _allowed_file(file.filename):
                return jsonify({
                    "error": f"Unsupported file type. "
                             f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
                }), 400
            image = Image.open(file.stream).convert("RGB")

        elif request.data:
            image = Image.open(BytesIO(request.data)).convert("RGB")

        else:
            return jsonify({
                "error": "No image provided. Send as 'image' file field "
                         "or raw bytes in the request body."
            }), 400

    except Exception as e:
        logger.error(f"Image decoding failed: {e}")
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    # ── Run inference ───────────────────────────────────────
    try:
        result = engine.predict(image, top_k=3)
        logger.info(
            f"Prediction: {result['predictions'][0]['category']} "
            f"({result['predictions'][0]['confidence']:.2%}) "
            f"| {result['latency_ms']}ms"
        )
        return jsonify(result), 200

    except Exception as e:
        logger.exception("Inference error")
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


# ── Error handlers ───────────────────────────────────────────

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Maximum size is 16 MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint not found.",
        "available_endpoints": {
            "GET  /health":  "API status check",
            "POST /predict": "Image classification",
        }
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


# ── Dev server ───────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
