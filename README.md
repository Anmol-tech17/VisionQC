# VisionQC — Automated PCB Defect Inspection

**MLOps Course Project — Status: Completed (Phases 1 through 6: Docker, Airflow, Industrial GUI, CI/CD, Prometheus & Grafana)**

An end-to-end machine learning system for automated detection of PCB (Printed Circuit Board) surface defects. The project features a YOLOv8s object detection champion model trained on real-world defects, tracked via MLflow, versioned via DVC, served dynamically through a containerized FastAPI REST API and modern industrial web dashboard, orchestrated via Apache Airflow, validated via GitHub Actions CI/CD, and continuously monitored via Prometheus and Grafana.

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
├── docs/                        # Project documentation (Cards, Reports, Demo Guide)
├── mlruns/                      # MLflow experiment tracking output
├── mlflow.db                    # MLflow SQLite backend
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Dataset

*Detailed dataset research and comparisons are available in `docs/DATASET_RESEARCH.md` and `docs/DATA_CARD.md`.*

**Researched Datasets:**
1. **PCB-Defect (Rashid 2025):** 230 real-world high-resolution images.
2. **Mixed PCB Defect Dataset (Mendeley):** 1,386 synthetic augmented images.

**Final Selected Dataset:**
The **PCB-Defect (Rashid 2025)** dataset was exclusively selected to train the final champion model due to its high authenticity and real-world defect representation, avoiding the biases of synthetic artifacts. The raw dataset is tracked via DVC.

---

## Model Experiments & Final Champion

*Detailed model comparisons are available in `docs/MODEL_CARD.md`.*

Through iterative experimentation managed by MLflow, several models were trained:
- **Baseline (YOLOv8n, 320px):** Poor recall due to tiny defect sizes.
- **Experiment A (YOLOv8n, 640px):** Massive improvement by increasing resolution.
- **Experiment B (YOLOv8n, 640px, V2 Dataset):** Tested synthetic data volume impact.

**Best / Final Champion Model:**
The **YOLOv8s (small)** architecture trained on the Rashid dataset at 640px (Phase 4A) was selected as the final champion.
- **Test mAP50:** 0.8137
- **Test mAP50-95:** 0.4294
- **Precision:** 0.8284 | **Recall:** 0.7863

---

## MLOps Pipeline & API (Phase 4B)

The project leverages **MLflow** for experiment tracking and model registration. The Champion model is tagged in the MLflow Model Registry as `VisionQC-Detector@champion`.

**FastAPI Service:**
The inference layer is a REST API that dynamically fetches the champion weights from MLflow on startup.

**Quick Start Demo:**
1. Install dependencies: `pip install -r requirements.txt`
2. View MLflow Dashboard: `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db` (Port 5000)
3. Start FastAPI Service: `python -m uvicorn src.visionqc.api.main:app --reload` (Port 8000)
4. Access Swagger UI: `http://localhost:8000/docs`

**API Endpoints:**
- `GET /health`: Returns service status and dynamically loaded MLflow model version.
- `POST /predict`: Accepts image uploads and returns a JSON payload of detected bounding boxes and classes.

---

## Course MLOps Requirements Status

| Component | Status | Verification Reference |
|---|---|---|
| Git + meaningful commits | Implemented ✅ | Git history (Phases 1–6) |
| Fixed seed | Implemented (seed=42) ✅ | `scripts/train.py` |
| Baseline model | Implemented (Phase 2) ✅ | YOLOv8n @ 320px |
| Candidate & Champion models | Implemented (Phase 4A) ✅ | YOLOv8s @ 640px (mAP50=0.8137) |
| MLflow tracking & registry | Implemented (Phase 3) ✅ | `mlflow.db` (`VisionQC-Detector@champion` v2) |
| DVC dataset versioning | Implemented (Phase 3) ✅ | `data/raw/rashid.dvc`, `mendeley.dvc` |
| FastAPI REST API (Swagger) | Implemented (Phase 4B) ✅ | `src/visionqc/api/main.py` (`/health`, `/predict`, `/docs`) |
| Industrial Web GUI | Implemented (Phase 6A) ✅ | `src/visionqc/api/static/index.html` |
| Docker Containerization | Implemented (Phase 5A) ✅ | `Dockerfile`, `docker-compose.yml` (`visionqc-api:latest`) |
| Airflow Orchestration | Implemented (Phase 5B) ✅ | `dags/visionqc_pipeline.py`, `docker-compose.airflow.yml` |
| GitHub Actions CI/CD | Implemented (Phase 6B) ✅ | `.github/workflows/ci.yml`, `docker.yml` |
| Prometheus & Grafana Monitoring | Implemented (Phase 6C) ✅ | `monitoring/`, `docker-compose.monitoring.yml` |
| Unit + integration tests | Implemented ✅ | `tests/test_api.py`, `tests/test_ci.py` |
| Model card & Data card | Implemented ✅ | `docs/MODEL_CARD.md`, `docs/DATA_CARD.md` |
| Project Overview & Demo Guide | Implemented ✅ | `docs/PROJECT_OVERVIEW.md`, `docs/DEMO_GUIDE.md` |
