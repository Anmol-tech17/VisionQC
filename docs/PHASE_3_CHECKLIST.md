# VisionQC — Phase 3 Compliance Checklist (DVC & MLflow)

## 1. DVC Implementation
- [x] **DVC Initialized**: Initialized in the repository (`.dvc/`).
- [x] **Local Remote Configured**: Configured `../dvc_storage` as the local remote to simulate an S3 bucket without network overhead.
- [x] **Rashid Dataset Tracked**: Added and pushed `data/raw/rashid` to DVC. Git ignores raw files but tracks `.dvc` files.
- [x] **Mendeley Dataset Tracked**: Downloaded Mendeley dataset (1741 synthetic images), stored in `data/raw/mendeley`, added to DVC, and pushed to remote.

## 2. MLflow Tracking Implementation
- [x] **Local SQLite Backend**: Configured MLflow to use `sqlite:///mlflow.db` to prevent filesystem lock crashes.
- [x] **Parameters Logged**: YOLOv8 hyperparameters (epochs, batch size, imgsz, seed, patience) tracked successfully in `train.py`.
- [x] **Metrics Logged**: Validation precision, recall, mAP50, and mAP50-95 logged successfully.
- [x] **Artifacts Logged**: Best model weights (`best.pt`), training run configuration (`training_run.json`), and evaluation metrics/graphs are pushed to MLflow artifacts.

## 3. Dataset Merging & Pipeline Isolation (V2 Dataset)
- [x] **Zero-Leakage Split**: Mendeley images (classes 0-5) were safely migrated directly into the `train` pipeline (`data/prepared/train`).
- [x] **Safe Class Remapping**: Mendeley's `missing hole` class (ID 0) safely remapped to class ID 6 (`missing_hole`) to keep it distinct from the original dataset's `missing_pad` (ID 0).
- [x] **Rashid Test Independence**: Original `val` and `test` splits contain exclusively real-world Rashid images.

## 4. Experiment Runs
- [x] **Experiment 0 (Baseline)**: Baseline YOLOv8n model trained on Rashid-only dataset. MLflow run backfilled safely.
- [x] **Experiment A (Candidate A)**: Placeholder YOLOv8s (Phase 2 model run).
- [x] **Experiment B (Candidate B)**: V2 YOLOv8n model trained on the expanded Rashid + Mendeley dataset (1,902 training images). Evaluated strictly on the held-out real-world test set. Logged and artifacted in MLflow.

**Status:** Phase 3 Complete.
