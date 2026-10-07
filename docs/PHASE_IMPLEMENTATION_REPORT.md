# VisionQC: Detailed Phasewise Implementation Report

This report documents the current implementation status of the VisionQC project up through Phase 4B.

## Phase 1: Project Initialization & Structure
- Set up the MLOps project repository structure.
- Created standard directories (`data/`, `notebooks/`, `scripts/`, `src/`, `tests/`, `docs/`, `configs/`).
- Initialized Git and `.gitignore`.
- Defined initial project rules and constraints.
- Selected and downloaded the raw PCB defect dataset (annotated in COCO format).

## Phase 2: Dataset Preparation & Baseline Model
- **Dataset Preparation**: Created `scripts/prepare_dataset.py` to convert COCO annotations to YOLO format and generate reproducible train/val/test splits (seed=42).
- **Dataset Audit**: Created `scripts/audit_dataset.py` to validate annotations and analyze defect sizes, finding that 96.3% of bounding boxes are smaller than 5% of the image size.
- **Baseline Training**: Trained a YOLOv8n model at 320px image size for 20 epochs.
  - Test mAP50: 0.1650
  - Precision: 0.2470
  - Recall: 0.1668
- **Experiment A**: Trained a YOLOv8n model at 640px image size for 40 epochs. Improved performance significantly over 320px due to tiny defect sizes.
- **Inference Pipeline**: Created foundational inference code (`model_loader.py`, `predictor.py`, `visualizer.py`) for YOLO models.

## Phase 3: DVC & MLflow Pipeline
- **DVC Integration**: Configured Data Version Control (DVC) for tracking large datasets and model artifacts outside Git.
- **MLflow Tracking**: Integrated MLflow (`mlruns/`, `mlflow.db`) to log metrics, parameters, and model artifacts for all training runs and experiments.
- **Model Registry**: Configured MLflow Model Registry to store trained models and manage their lifecycle (e.g., using `champion` aliases).
- **Refactoring**: Updated training and evaluation scripts to automatically push metrics and artifacts to the MLflow backend.

## Phase 4A: Model Improvement
- **Objective**: Improve model performance over the Phase 2 baseline.
- **Baseline (from Phase 2 Experiment A / earlier)**:
  - Model: YOLOv8n
  - Image Size: 640px
  - Epochs: 40
  - Test mAP50: **0.7527**
  - Test mAP50-95: **0.3610**
  - CPU Latency: **164.9 ms**
- **Candidate Model**:
  - Model: YOLOv8s
  - Image Size: 640px
  - Epochs: 40
  - Test mAP50: **0.8137**
  - Test mAP50-95: **0.4294**
  - Precision: **0.8284**
  - Recall: **0.7863**
  - CPU Latency: **346.0 ms**
- **Decision**: YOLOv8s was selected as the new champion because it significantly improved the held-out test metrics.
- **Model Registry Update**: The YOLOv8s model was registered as `VisionQC-Detector` in MLflow, and the `champion` alias was assigned to Version 2.

## Phase 4B: FastAPI MLflow Inference Service
- **Objective**: Build a robust REST API that serves predictions using the latest champion model from the MLflow registry.
- **Implementation**:
  - Created `src/visionqc/api/main.py` using FastAPI.
  - Implemented application lifespan events to fetch and load the MLflow model on startup.
  - **MLflow Champion Loading**: The service queries `models:/VisionQC-Detector@champion` dynamically.
  - **Workaround**: Due to a bug in the MLflow PyFunc wrapper's `.tojson()` implementation, the API downloads the raw weights artifact (`best.pt`) and loads it using the native Ultralytics YOLO class.
- **Endpoints**:
  - `GET /health`: Returns service health and the currently loaded MLflow model version.
  - `POST /predict`: Accepts an uploaded image file, processes it, and returns a JSON payload containing predicted classes, confidences, and bounding boxes.
- **Validation**:
  - Tested with real PCB images.
  - Verified graceful handling of invalid or non-image inputs.
  - Confirmed the API dynamically loads Version 2 (the current champion).
