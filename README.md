# 🛒 Smart E-commerce Product Classifier

A production-ready deep learning microservice that automatically classifies e-commerce product images into categories using **ResNet50 transfer learning**, served via a **Flask REST API** and containerized with **Docker**.

### 📊 Achieved Results

| Metric | Score |
|---|---|
| **Validation Accuracy** | **99.01%** |
| **Test Accuracy** | **98.98%** |
| **Macro F1 Score** | **0.987** |
| **Inference Latency** | **~350ms** (CPU) |
| **Dataset** | 44,310 images · 4 categories |

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **Transfer Learning** | Fine-tuned ResNet50 pretrained on ImageNet |
| 🚀 **Fast Inference** | < 500 ms per image |
| 📊 **Top-3 Predictions** | Returns confidence scores with auto-tag / human-review flag |
| 🔄 **Active Learning Ready** | Low-confidence predictions flagged for review |
| 🐳 **Docker Deployment** | Production-ready with Gunicorn WSGI server |
| 📈 **Full Evaluation Suite** | Accuracy, Precision, Recall, F1, Confusion Matrix |
| 📥 **Automated Data Pipeline** | One-command Kaggle dataset download & preprocessing |

---

## 📁 Project Structure

```
smart-product-classifier/
├── data/                        # Image dataset (ImageFolder layout)
│   ├── Apparel/         (21,395)
│   ├── Accessories/     (11,289)
│   ├── Footwear/         (9,222)
│   └── Personal Care/    (2,404)
├── models/                      # Trained model & metrics
│   ├── best_model.pth
│   ├── training_metrics.json
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── misclassified/
├── src/
│   ├── download_and_prep_data.py  # Automated Kaggle data pipeline
│   ├── dataset.py               # Data loading & augmentation
│   ├── model.py                 # ResNet50 architecture
│   ├── train.py                 # Training pipeline
│   ├── evaluate.py              # Evaluation & metrics
│   └── inference.py             # Inference engine
├── api/
│   └── app.py                   # Flask REST API
├── docker/
│   └── Dockerfile               # Production container
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/smart-product-classifier.git
cd smart-product-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

**Automated (recommended):** Download and preprocess the Kaggle fashion dataset with one command:

```bash
# Requires Kaggle API key at ~/.kaggle/kaggle.json
python src/download_and_prep_data.py
```

This automatically downloads `paramaggarwal/fashion-product-images-small` (44K images), organizes them into ImageFolder layout, and prunes minority classes (< 500 images).

**Manual:** Alternatively, organize your own product images in `data/` — one sub-folder per category:

```
data/
├── Apparel/
│   ├── img_001.jpg
│   └── ...
├── Footwear/
├── Accessories/
└── ...
```

> **Tip:** Aim for 500+ images per category for best results. The pipeline automatically handles the 70/15/15 train-val-test split.

### 3. Train the Model

```bash
python src/train.py
```

**Configurable via CLI:**
```bash
python src/train.py <data_dir> <num_epochs>
# Example:
python src/train.py ./data 20
```

Training outputs:
- `models/best_model.pth` — Best checkpoint
- `models/training_metrics.json` — Loss/accuracy history
- `models/training_curves.png` — Loss & accuracy plots

### 4. Evaluate

```bash
python src/evaluate.py
```

Produces:
- Test accuracy, macro Precision / Recall / F1
- `models/confusion_matrix.png`
- `models/misclassified/` — Misclassified image examples

### 5. Run the API

**Development:**
```bash
python api/app.py
```

**Production (Gunicorn):**
```bash
gunicorn --bind 0.0.0.0:5000 --workers 4 api.app:app
```

---

## 📡 API Reference

### `GET /health`

Returns API and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "models/best_model.pth",
  "version": "1.0.0"
}
```

### `POST /predict`

Classify a product image. Accepts `multipart/form-data` with an `image` field.

**Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@sample_shirt.jpg"
```

**Response:**
```json
{
  "predictions": [
    {"category": "Apparel", "confidence": 1.0},
    {"category": "Accessories", "confidence": 0.0},
    {"category": "Footwear", "confidence": 0.0}
  ],
  "requires_review": false,
  "latency_ms": 356.8
}
```

| Field | Description |
|---|---|
| `predictions` | Top-3 categories with confidence scores |
| `requires_review` | `true` if top confidence < 0.90 (flags for human review) |
| `latency_ms` | End-to-end inference time in milliseconds |

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -f docker/Dockerfile -t product-classifier .

# Run the container
docker run -p 5000:5000 product-classifier

# With external model volume
docker run -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -e MODEL_PATH=/app/models/best_model.pth \
  product-classifier
```

---

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/best_model.pth` | Path to trained model checkpoint |

---

## 🏗️ Tech Stack

- **PyTorch** + **Torchvision** — Deep learning backbone
- **Flask** — REST API framework
- **Gunicorn** — Production WSGI server
- **Docker** — Containerization
- **scikit-learn** — Evaluation metrics
- **matplotlib** + **seaborn** — Visualization
- **Kaggle API** + **pandas** — Automated data ingestion

---

## 📋 Business Logic

```
┌─────────────────────────────────────────────────────────┐
│  Image Upload                                           │
│       │                                                 │
│       ▼                                                 │
│  Preprocessing (Resize → Normalize)                     │
│       │                                                 │
│       ▼                                                 │
│  ResNet50 Inference → Top-3 Predictions                 │
│       │                                                 │
│       ├── Confidence ≥ 0.90 → ✅ Auto-tag product       │
│       │                                                 │
│       └── Confidence < 0.90 → 🔍 Flag for human review  │
│                                (Active Learning)        │
└─────────────────────────────────────────────────────────┘
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
