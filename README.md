# VisionQC — Automated PCB Defect Inspection

**MLOps Course Project — Phase 2**

An end-to-end machine learning system for automated detection of PCB (Printed Circuit Board) surface defects using real YOLOv8n object detection inference.

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

The system provides a FastAPI REST API and browser-based UI for real-time defect inspection.

---

## Repository Structure

```
VisionQC/
├── configs/
│   ├── settings.py                # Centralised path + threshold config
│   ├── training.yaml              # Baseline training configuration
│   └── training_exp_a.yaml        # Experiment A: 640px + PCB augmentation
│
├── data/
│   └── prepared/                  # YOLO-format dataset (generated, not committed)
│       ├── train/images/, train/labels/
│       ├── val/images/, val/labels/
│       ├── test/images/, test/labels/
│       └── dataset.yaml
│
├── artifacts/
│   └── models/
│       ├── pcb_yolov8n_baseline/  # Baseline 320px model
│       └── pcb_yolov8n_640/       # Experiment A 640px model (final)
│
├── scripts/
│   ├── prepare_dataset.py         # COCO → YOLO conversion + split
│   ├── audit_dataset.py           # Dataset quality audit
│   ├── train.py                   # Training launcher (supports --config)
│   └── evaluate.py                # Test-split evaluation (supports --model --name)
│
├── src/visionqc/
│   ├── api/
│   │   ├── main.py                # FastAPI app
│   │   └── static/index.html      # Browser UI
│   ├── data/
│   │   ├── dataset_prep.py        # COCO→YOLO converter class
│   │   └── validator.py           # Dataset validator
│   └── inference/
│       ├── model_loader.py        # Thread-safe model singleton
│       ├── predictor.py           # YOLOPredictor + GTAnnotationLookup
│       └── visualizer.py          # Bounding box drawing
│
├── tests/
│   └── test_api.py                # 20 API + unit tests
│
├── docs/
│   ├── DATA_CARD.md               # Dataset documentation
│   ├── MODEL_CARD.md              # Model documentation
│   └── PHASE_2_REPORT.md          # Full Phase 2 technical report
│
├── reports/
│   ├── dataset_report.md          # Dataset preparation report
│   ├── dataset_audit.json         # Quality audit results
│   ├── training_experiments.md    # Experiment comparison table
│   ├── training_run_training.json          # Baseline training record
│   ├── training_run_training_exp_a.json    # Exp A training record
│   ├── evaluation_results_baseline.json   # Baseline test metrics
│   └── evaluation_results_exp_a.json      # Exp A test metrics
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

The raw dataset is **not committed to Git** (too large). Store it locally at:

```
MLOPS CP/PCB-Defect An Annotated Dataset for Surface Defect/
  PCB-Defect An Annotated Dataset for Surface Defect/
    PCB_Defect/PCB_Defect/
      annotation/_annotations.coco.json
      images/   (230 JPEG images)
```

Dataset will be versioned with DVC in a later phase.

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
Hardware:   Intel Core i3-8130U (CPU only)
```

> **Important:** All scripts must be run with Python 3.13.
> In PowerShell use: `& "C:\Program Files\Python313\python.exe"`

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare dataset

```bash
& "C:\Program Files\Python313\python.exe" scripts/prepare_dataset.py
```

Output: `data/prepared/` — YOLO dataset with train/val/test splits (seed=42)

### 3. Audit dataset (optional but recommended)

```bash
& "C:\Program Files\Python313\python.exe" scripts/audit_dataset.py
```

Output: `reports/dataset_audit.json`

### 4. Train baseline model

```bash
& "C:\Program Files\Python313\python.exe" scripts/train.py
```

Output: `artifacts/models/pcb_yolov8n/weights/best.pt`

### 5. Train Experiment A (640px)

```bash
& "C:\Program Files\Python313\python.exe" scripts/train.py --config configs/training_exp_a.yaml
```

Output: `artifacts/models/pcb_yolov8n_640/weights/best.pt`

### 6. Evaluate on test split

```bash
# Evaluate Experiment A
& "C:\Program Files\Python313\python.exe" scripts/evaluate.py \
    --model artifacts/models/pcb_yolov8n_640/weights/best.pt \
    --name exp_a
```

Output: `reports/evaluation_results_exp_a.json`, `reports/evaluation_results_exp_a.md`

### 7. Run tests

```bash
& "C:\Program Files\Python313\python.exe" -m pytest tests/ -v
```

### 8. Start the API server

```bash
& "C:\Program Files\Python313\python.exe" -m uvicorn src.visionqc.api.main:app --reload --host 0.0.0.0 --port 8000
```

Open: http://localhost:8000

---

## Training Configuration

All training parameters live in `configs/training.yaml` (baseline) or `configs/training_exp_a.yaml` (Experiment A). To override via CLI:

```bash
# Override epochs and image size
& "C:\Program Files\Python313\python.exe" scripts/train.py --epochs 50 --imgsz 640
```

| Parameter | Baseline | Exp A | Note |
|---|---|---|---|
| `seed` | 42 | 42 | Fixed for reproducibility |
| `model` | yolov8n.pt | yolov8n.pt | Pretrained COCO weights |
| `image_size` | 320 | 640 | 640 recommended (96% small objects) |
| `epochs` | 20 | 40 | — |
| `patience` | 5 | 10 | Early stopping patience |
| `batch_size` | 4 | 4 | CPU constraint |
| `workers` | 0 | 0 | Avoids multiprocessing overhead |
| `mosaic` | default | 0.0 | OFF: crops destroy tiny defect context |
| `degrees` | default | 0.0 | OFF: PCB orientation matters |

---

## Experiments

### Dataset Audit Finding

**96.3% of all bounding boxes are smaller than 5% of image width/height.**

| Image size | Median defect (px) |
|---|---|
| 320 px | ~9 × 9 |
| 640 px | ~19 × 19 |

This is the root cause of poor recall on `mouse_bite`, `spur`, and `missing_pad`.

### Baseline Results (test split)

| Metric | Value |
|---|---|
| Precision | 0.2470 |
| Recall | 0.1668 |
| mAP@0.5 | 0.1650 |
| mAP@0.5:0.95 | 0.0670 |
| Mean latency | 128 ms |
| P95 latency | 267 ms |

| Class | AP@0.5 |
|---|---|
| spurious_copper | 0.4621 (**best**) |
| open_circuit | 0.2422 |
| short | 0.1987 |
| missing_pad | 0.0510 |
| spur | 0.0221 |
| mouse_bite | 0.0141 (**worst**) |

### Experiment A Results

> *Run after training completes:*
> `& "C:\Program Files\Python313\python.exe" scripts/evaluate.py --model artifacts/models/pcb_yolov8n_640/weights/best.pt --name exp_a`

---

## API Reference

### GET /health

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "pcb_yolov8n_640",
  "confidence_threshold": 0.25,
  "dataset_valid": true,
  "uptime_seconds": 42.1
}
```

### POST /predict

Request: multipart/form-data, field `file` = JPEG or PNG image

Response:
```json
{
  "detections": [
    {"class_name": "short", "class_id": 3, "confidence": 0.87, "bbox": [x,y,w,h], "source": "yolo_model"}
  ],
  "count": 1,
  "annotated_image": "<base64-JPEG>",
  "prediction_type": "yolo_model",
  "confidence_threshold": 0.25
}
```

Returns 503 if model not loaded. Returns 422 for non-image uploads.

### GET /validate

Returns dataset validation results.

---

## MLOps Pipeline (Current Phase)

```
Raw COCO dataset
     ↓
scripts/prepare_dataset.py  (COCO→YOLO, reproducible split, seed=42)
     ↓
data/prepared/dataset.yaml
     ↓
scripts/train.py --config configs/training_exp_a.yaml
     ↓
artifacts/models/pcb_yolov8n_640/weights/best.pt
     ↓
scripts/evaluate.py --name exp_a   (test split only)
     ↓
reports/evaluation_results_exp_a.json
     ↓
FastAPI /predict endpoint
     ↓
Browser UI
```

---

## Reproducibility

| Item | Value |
|---|---|
| Random seed | 42 (all configs) |
| Python | 3.13.0 |
| PyTorch | 2.14.0+cpu |
| Ultralytics | 8.4.160 |
| Dataset split | Image-level, seed=42, 70/15/15 |

The `data/prepared/` directory is deterministic — running `prepare_dataset.py` with seed=42 always produces the same split.

---

## Known Limitations

1. **Small dataset**: 230 images is insufficient for production-grade object detection
2. **Small objects**: Even at 640px, the smallest defects are ~3px — very hard to detect reliably
3. **CPU training**: Limited to ~100 min/experiment; prevents extensive hyperparameter search
4. **Single domain**: Dataset comes from one type of PCB; generalisation is uncertain
5. **No negatives**: Model has not seen defect-free PCBs; false positive rate unknown

---

## Course MLOps Requirements Status

| Component | Status |
|---|---|
| Git + meaningful commits | Implemented |
| Fixed seed | seed=42 everywhere |
| Baseline model | Phase 2 ✅ |
| Candidate models (≥2) | Baseline + Exp A ✅ (more in MLflow phase) |
| MLflow tracking | Planned Phase 3 |
| MLflow model registry | Planned Phase 3 |
| Quality gate | Thresholds TBD after Exp A |
| DVC dataset versioning | Planned Phase 3 |
| Airflow orchestration | Planned Phase 3 |
| FastAPI REST API | Implemented ✅ |
| Docker | Planned Phase 3 |
| GitHub Actions CI/CD | Planned Phase 3 |
| Unit + integration tests | 20 tests ✅ |
| Model card | docs/MODEL_CARD.md ✅ |
| Data card | docs/DATA_CARD.md ✅ |
| Monitoring | Planned Phase 3 |
| Explainability | Planned Phase 3 |
| AWS deployment | Planned Phase 3 |
