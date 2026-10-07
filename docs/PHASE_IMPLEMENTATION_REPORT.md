# VisionQC: Detailed Phasewise Implementation Report

This report is the single authoritative phase-by-phase implementation record for the VisionQC project.

---

## Phase 1: Project Initialization & Structure
* **Objective:** Establish the foundation for the MLOps project.
* **Work Performed:** 
  - Created standard directory structures (`data/`, `notebooks/`, `scripts/`, `src/`, `tests/`, `docs/`, `configs/`).
  - Initialized Git repository and configured `.gitignore`.
  - Researched and selected the initial raw PCB defect dataset (Rashid 2025, annotated in COCO format).
* **Important Decisions:** Opted for a modular script-based architecture rather than relying solely on Jupyter notebooks to prepare for API deployment.
* **Current Status:** Complete.

---

## Phase 2: Dataset Preparation & Baseline Model
* **Objective:** Build the ML inference layer and establish a performance baseline.
* **Work Performed:**
  - **Data Pipeline:** Created `scripts/prepare_dataset.py` to convert COCO annotations to YOLO format, generating reproducible train/val/test splits (seed=42).
  - **Dataset Audit:** Executed `scripts/audit_dataset.py`, discovering that 96.3% of defects are smaller than 5% of the image size.
  - **Baseline Training:** Trained YOLOv8n at 320px for 20 epochs. Performance was extremely poor (mAP50: 0.1650) because defects were too small to detect.
  - **Experiment A:** Increased resolution to 640px and disabled mosaic augmentation. Performance jumped drastically.
  - **Inference Code:** Created foundational Python inference classes (`model_loader.py`, `predictor.py`, `visualizer.py`).
* **Important Decisions:** Identified 640px as the minimum viable resolution for this specific dataset based on bounding-box statistical analysis.
* **Results:** Established a working YOLO pipeline with significant metric improvements through resolution tuning.
* **Current Status:** Complete.

---

## Phase 3: DVC & MLflow Pipeline
* **Objective:** Implement data versioning and experiment tracking.
* **Work Performed:**
  - **DVC Integration:** Initialized Data Version Control (`.dvc/`) to track datasets and large model artifacts outside of Git, using a local remote (`../dvc_storage`).
  - **MLflow Integration:** Configured MLflow with a local SQLite backend (`mlflow.db`). Refactored training/evaluation scripts to auto-log parameters, metrics, and artifacts.
  - **Experiment B:** Researched the synthetic Mendeley dataset (1,386 images), merged it with the Rashid dataset to create a "V2 Dataset", and trained a model to observe the impact of synthetic augmentation.
  - **Model Registry:** Activated the MLflow Model Registry to manage lifecycle stages via aliases.
* **Important Decisions:** Rejected the synthetic Mendeley dataset for final production use due to domain-gap concerns, strictly utilizing MLflow to backfill and track all experiments.
* **Current Status:** Complete.

---

## Phase 4A: Model Improvement
* **Objective:** Train and register a final, highly performant "Champion" model using the tracked MLOps pipeline.
* **Work Performed:**
  - Scaled the architecture from YOLOv8n to **YOLOv8s**.
  - Trained on the pure, real-world Rashid dataset at 640px for 40 epochs.
* **Results / Metrics:**
  - Held-out Test mAP50: **0.8137** (Improved from 0.7527)
  - Held-out Test mAP50-95: **0.4294**
  - Precision: **0.8284** | Recall: **0.7863**
  - CPU Latency: 346.0 ms
* **Important Decisions:** YOLOv8s was officially selected as the project Champion.
* **Outputs:** The model was registered in MLflow as `VisionQC-Detector`, and the `champion` alias was assigned to Version 2.
* **Current Status:** Complete.

---

## Phase 4B: FastAPI MLflow Inference Service
* **Objective:** Build a REST API that serves predictions dynamically using the MLflow champion model.
* **Work Performed:**
  - Created `src/visionqc/api/main.py` using FastAPI.
  - Programmed lifespan startup events to query MLflow for `models:/VisionQC-Detector@champion` and download the artifact dynamically.
  - Implemented `/health` and `/predict` endpoints.
  - Tested API endpoint robustness, including graceful handling of invalid file uploads.
* **Problems & Solutions:** Discovered a bug in the MLflow PyFunc wrapper's `.tojson()` implementation. Solved it by extracting the raw `best.pt` weights artifact and loading it directly via the native Ultralytics YOLO class.
* **Important Outputs:** A fully functional, Swagger-documented REST API ready for Dockerization.
* **Current Status:** Complete.
