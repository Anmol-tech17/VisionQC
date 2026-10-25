# VisionQC: Detailed Phasewise Implementation Report

This is the single authoritative phasewise implementation record for VisionQC.
Updated through **Phase 5C (Integration)**.

---

## Phase 1: Project Initialization & Structure

* **Objective:** Establish the MLOps project foundation.
* **Work Performed:** Standard directory structure, Git init, `.gitignore`, dataset selection (Rashid 2025).
* **Status:** ✅ Complete (commit `cebfc64`)

---

## Phase 2: Dataset Preparation & Baseline Model

* **Objective:** Build the ML inference layer and establish a performance baseline.
* **Dataset:** PCB-Defect (Rashid 2025) — 230 real-world images, 6 defect classes, COCO→YOLO converted.
* **Key Finding:** 96.3% of bounding boxes < 5% of image size → 640px resolution required.
* **Baseline (YOLOv8n, 320px, 20 epochs):** mAP50=0.1650, Precision=0.2470, Recall=0.1668
* **Experiment A (YOLOv8n, 640px, 40 epochs):** mAP50=0.7527, Precision=0.8192, Recall=0.6954
* **Inference Code:** `model_loader.py`, `predictor.py`, `visualizer.py` created.
* **Status:** ✅ Complete (commit `3b74097`)

---

## Phase 3: DVC & MLflow Pipeline

* **Objective:** Data versioning and experiment tracking.
* **DVC:** Initialized with local remote at `../../dvc_storage`. Rashid and Mendeley datasets tracked.
* **MLflow:** SQLite backend (`mlflow.db`), experiments tracked for all runs.
* **Experiment B:** Merged Rashid + Mendeley (V2 Dataset, 1,902 training images). Rejected for final model due to synthetic domain gap.
* **Status:** ✅ Complete (commit `5d8d52c`)

---

## Phase 4A: Model Improvement

* **Objective:** Train and register a final Champion model.
* **Model:** YOLOv8s (small) architecture on pure Rashid dataset, 640px, 40 epochs.
* **Test Results:**
  * mAP50: **0.8137** | mAP50-95: **0.4294** | Precision: **0.8284** | Recall: **0.7863**
  * CPU Latency: 346.0 ms mean
* **MLflow:** Registered as `VisionQC-Detector`, `champion` alias → Version 2
* **Status:** ✅ Complete (commit `1d909fb`)

---

## Phase 4B: FastAPI MLflow Inference Service

* **Objective:** REST API that dynamically serves the MLflow champion model.
* **Implementation:** FastAPI + lifespan startup that queries `models:/VisionQC-Detector@champion`.
* **Workaround:** MLflow PyFunc `.tojson()` bug → raw `best.pt` loaded via Ultralytics YOLO directly.
* **Endpoints:** `GET /health` (model version), `POST /predict` (bounding boxes + classes JSON)
* **Status:** ✅ Complete (commit `9c103aa`)

---

## Phase 5A: Docker (Containerization)

* **Objective:** Package the Phase 4B inference service in a reproducible Docker container.
* **Docker Engine:** Ubuntu-D WSL2 distribution, Docker 29.9.2, Compose v5.6.1 (D: drive).
* **Files Created:** `Dockerfile`, `.dockerignore`, `docker-compose.yml`
* **Base image:** `python:3.12-slim` + CPU-only PyTorch
* **MLflow:** `mlflow.db` and `mlruns/` bind-mounted from C: (project location unchanged)
* **API accessible at:** `http://localhost:8000` from Windows
* **Image name:** `visionqc-api:latest`
* **Status:** ✅ Complete (commit `00c48ca`)

---

## Phase 5B: Airflow (Pipeline Orchestration)

* **Objective:** Orchestrate the VisionQC training pipeline via Apache Airflow.
* **Airflow version:** `apache/airflow:2.10.3-python3.12`, SequentialExecutor, SQLite metadata DB on D:
* **DAG:** `visionqc_training_pipeline` — 6 tasks: validate → prepare → train → evaluate → compare → record
* **Safety:** Smoke-test mode by default (1 epoch). Champion model never auto-promoted by DAG.
* **Files Created:** `dags/visionqc_pipeline.py`, `docker-compose.airflow.yml`
* **Airflow UI:** `http://localhost:8080` (admin/admin)
* **Status:** ✅ Complete (commit `eaf9e26`)

---

## Phase 5C: Integration & Regression

* **Objective:** Verify Docker, Airflow, and existing components work together.
* **Confirmed:** API (port 8000) and Airflow (port 8080) run independently and concurrently without port conflicts.
* **Confirmed:** Project location unchanged (remains on C:).
* **Confirmed:** Docker/Airflow persistent data stored on D: (Ubuntu-D virtual disk + `/mnt/d/visionqc-airflow/`).
* **Confirmed:** MLflow champion (`VisionQC-Detector@champion`, Version 2) and DVC config unmodified.
* **Refinement:** Increased container memory limit in `docker-compose.yml` to 3 GB to provide PyTorch runtime headroom.
* **Status:** ✅ Complete

---

## MLOps Course Requirements Status

| Component | Status |
|---|---|
| Git + meaningful commits | ✅ Implemented |
| Fixed seed (42) | ✅ Implemented |
| Baseline model | ✅ Phase 2 |
| Candidate models | ✅ Phase 4A |
| MLflow tracking & registry | ✅ Phase 3 |
| DVC dataset versioning | ✅ Phase 3 |
| FastAPI REST API | ✅ Phase 4B |
| Docker (containerization) | ✅ **Phase 5A** |
| Airflow orchestration | ✅ **Phase 5B** |
| Unit + integration tests | ✅ Implemented |
| CI/CD | Planned Phase 6 |
| Monitoring / Drift detection | Planned Phase 6 |
| Cloud deployment | Planned Phase 6 |
