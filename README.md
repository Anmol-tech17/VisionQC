# VisionQC — Automated PCB Defect Inspection

**MLOps Course Project — Phase 4B: FastAPI MLflow Inference Service**

An end-to-end machine learning system for automated detection of PCB (Printed Circuit Board) surface defects using real YOLOv8s object detection inference, tracked via MLflow and served via FastAPI.

---

## Project Overview

VisionQC detects six categories of PCB manufacturing defects:

| ID | Class | Description |
|---|---|---|
| 0 | `missing_pad` | Solder pad missing from expected location |
| 1 | `mouse_bite` | Small notch on PCB edge |
| 2 | `open_circuit` | Break in a conductive trace |
| 3 | `short` | Unintended connection between traces |
| 4 | `spur` | Unwanted metal protrusion |
| 5 | `spurious_copper` | Copper appearing where it should not |

The system provides a FastAPI REST API for real-time defect inspection, with model management handled by MLflow.

---

## Repository Structure

```
VisionQC/
├── configs/                     # Training configurations
├── data/                        # Prepared YOLO datasets (generated)
├── artifacts/                   # Model artifacts and weights (DVC)
├── scripts/                     # Data preparation, training, evaluation scripts
├── src/visionqc/
│   ├── api/                     # Phase 4B: FastAPI MLflow Service
│   │   ├── main.py              # Application entrypoint
│   │   └── README.md
│   ├── data/                    # Dataset processing
│   └── inference/               # Inference utilities
├── tests/                       # Unit and API tests
├── docs/                        # Project documentation & DEMO_GUIDE
├── mlruns/                      # MLflow experiment tracking output
├── mlflow.db                    # MLflow SQLite backend
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

The raw dataset is **not committed to Git** (too large). It consists of 230 annotated PCB images in COCO format.

| Split | Images | Annotations |
|---|---|---|
| Train | 161 | 1,194 |
| Val | 34 | 243 |
| Test | 35 | 267 |
| **Total** | **230** | **1,704** |

---

## Environment

```
Python:     3.13.0
PyTorch:    2.14.0+cpu
Ultralytics: 8.4.160
Hardware:   CPU only
```

> **Important:** All scripts must be run with Python 3.13.

---

## Quick Start (Phase 4B Demo)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. View MLflow Dashboard
```bash
& "C:\Program Files\Python313\python.exe" -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open `http://localhost:5000` to view the trained models, experiments, and the `VisionQC-Detector` in the Model Registry.

### 3. Start the FastAPI Service
```bash
& "C:\Program Files\Python313\python.exe" -m uvicorn src.visionqc.api.main:app --reload --host 0.0.0.0 --port 8000
```
Open the interactive Swagger UI at: `http://localhost:8000/docs`

---

## Model Performance (Phase 4A)

The current **Champion** model deployed to the API is **YOLOv8s** (Version 2).

| Metric | Phase 2 Baseline (YOLOv8n) | Phase 4A Champion (YOLOv8s) |
|---|---|---|
| Image Size | 640px | 640px |
| Epochs | 40 | 40 |
| mAP@0.5 | 0.7527 | **0.8137** |
| mAP@0.5:0.95 | 0.3610 | **0.4294** |
| Precision | - | **0.8284** |
| Recall | - | **0.7863** |
| CPU Latency | 164.9 ms | 346.0 ms |

YOLOv8s was selected as it substantially improved overall detection metrics on the tiny PCB defects.

---

## API Reference (FastAPI)

### `GET /health`
Returns the status of the API and dynamically reports the version of the MLflow champion model currently loaded.

```json
{
  "status": "ok",
  "model_version": 2
}
```

### `POST /predict`
Uploads an image to receive defect predictions.

**Request:** `multipart/form-data`, field `file` containing the image.

**Example using `curl`:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/prepared/test/images/pcb_defect_test.jpg"
```

**Response (200 OK):**
```json
{
  "predictions": [
    {
      "name": "short",
      "class": 3,
      "confidence": 0.895,
      "box": {
        "x1": 150.5,
        "y1": 200.0,
        "x2": 175.2,
        "y2": 220.8
      }
    }
  ]
}
```

---

## MLOps Pipeline Status

```
Raw COCO dataset
     ↓
Dataset Prep & Split (Reproducible, seed=42)
     ↓
DVC Versioning & MLflow Tracking (Phase 3)
     ↓
Model Training & Experimentation (Phase 4A: YOLOv8s)
     ↓
MLflow Model Registry (Champion Alias)
     ↓
FastAPI Dynamic Inference Service (Phase 4B)
```

## Course MLOps Requirements Status

| Component | Status |
|---|---|
| Git + meaningful commits | Implemented ✅ |
| Fixed seed | seed=42 everywhere ✅ |
| Baseline model | Phase 2 ✅ |
| Candidate models | Phase 4A ✅ |
| MLflow tracking | Phase 3 ✅ |
| MLflow model registry | Phase 3 ✅ |
| DVC dataset versioning | Phase 3 ✅ |
| FastAPI REST API | Phase 4B ✅ |
| Docker | Planned Phase 5A |
| Airflow orchestration | Planned Phase 5B |
| Unit + integration tests | Implemented ✅ |
| Model card & Data card | Implemented ✅ |
