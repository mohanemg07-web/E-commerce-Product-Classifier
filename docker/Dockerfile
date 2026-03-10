# ============================================================
# Smart E-commerce Product Classifier — Production Dockerfile
# ============================================================
# Build:  docker build -f docker/Dockerfile -t product-classifier .
# Run:    docker run -p 5000:5000 -v ./models:/app/models product-classifier
# ============================================================

# ── Stage: Runtime ──────────────────────────────────────────
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies (minimal)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

WORKDIR /app

# ── Install Python dependencies first (layer caching) ───────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──────────────────────────────────
COPY src/     src/
COPY api/     api/
COPY models/  models/

# Ensure the models directory exists
RUN mkdir -p models && chown -R appuser:appuser /app

USER appuser

# ── Expose and launch ──────────────────────────────────────
EXPOSE 5000

# Gunicorn config: 4 workers, 120s timeout, bind to 0.0.0.0:5000
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "4", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "api.app:app"]
