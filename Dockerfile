# VisionQC — Dockerfile (Phase 5A)
# Containerises the FastAPI MLflow inference service.
#
# Build:
#   docker build -t visionqc-api:latest .
#
# Run (standalone):
#   docker run -p 8000:8000 \
#     -v /path/to/mlflow.db:/app/mlflow.db:ro \
#     -v /path/to/mlruns:/app/mlruns:ro \
#     visionqc-api:latest
#
# The container connects to the local MLflow SQLite backend at /app/mlflow.db
# and downloads the VisionQC-Detector@champion artifact at startup.

# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.12 (Ubuntu 24.04 matches WSL, stable, avoids Python 3.13 edge-cases)
FROM python:3.12-slim

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL maintainer="VisionQC Team" \
      description="VisionQC FastAPI PCB Defect Inference Service" \
      version="1.0.0"

# ── OS dependencies ───────────────────────────────────────────────────────────
# libGL / libglib: required by OpenCV (headless still needs libglib)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy only the requirements file first (layer-cache friendly).
COPY requirements.txt ./

# Install CPU-only torch explicitly before ultralytics to avoid GPU wheel pulls.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        torch==2.4.1+cpu \
        torchvision==0.19.1+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
# Copy source modules needed by the FastAPI app.
COPY src/ ./src/
COPY configs/ ./configs/

# ── Non-root user & cross-platform artifact path compatibility ────────────────
# MLflow records Windows host paths (file:C:/Users/...) in sqlite. Create symlinks
# mapping relative and root-level Windows path targets to /app/mlruns so artifact resolution works.
RUN mkdir -p '/app/C:/Users/dell/OneDrive/Desktop/MLOPS CP/VisionQC' '/C:/Users/dell/OneDrive/Desktop/MLOPS CP/VisionQC' \
    && ln -s /app/mlruns '/app/C:/Users/dell/OneDrive/Desktop/MLOPS CP/VisionQC/mlruns' \
    && ln -s /app/mlruns '/C:/Users/dell/OneDrive/Desktop/MLOPS CP/VisionQC/mlruns' \
    && useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app '/C:'
USER appuser

# ── Runtime environment variables ─────────────────────────────────────────────
# MLflow tracking URI inside the container.
# mlflow.db and mlruns/ are expected to be volume-mounted at /app/.
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "src.visionqc.api.main:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
