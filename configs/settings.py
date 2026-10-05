"""
VisionQC — Application Settings (Phase 2)
-------------------------------------------
All tunable values are read from environment variables with sensible defaults.
No absolute paths are hard-coded in source.

Environment variables
---------------------
Dataset
    VISIONQC_DATASET_DIR          Root of the PCB_Defect dataset directory
    VISIONQC_ANNOTATION_FILE      Full path to _annotations.coco.json
    VISIONQC_IMAGES_DIR           Full path to images/

Prepared data (after running scripts/prepare_dataset.py)
    VISIONQC_PREPARED_DIR         Root of the YOLO-format prepared dataset

Model
    VISIONQC_MODEL_PATH           Path to best.pt (trained model)
    VISIONQC_CONFIDENCE_THRESHOLD Minimum detection confidence (default 0.25)
    VISIONQC_IOU_THRESHOLD        NMS IoU threshold (default 0.45)
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent      # configs/
PROJECT_ROOT = _HERE.parent                   # VisionQC/

# ---------------------------------------------------------------------------
# Raw dataset
# ---------------------------------------------------------------------------
_DEFAULT_DATASET_DIR = (
    PROJECT_ROOT.parent
    / "PCB-Defect An Annotated Dataset for Surface Defect"
    / "PCB-Defect An Annotated Dataset for Surface Defect"
    / "PCB_Defect"
    / "PCB_Defect"
)

DATASET_DIR: Path = Path(
    os.environ.get("VISIONQC_DATASET_DIR", str(_DEFAULT_DATASET_DIR))
)
ANNOTATION_FILE: Path = Path(
    os.environ.get(
        "VISIONQC_ANNOTATION_FILE",
        str(DATASET_DIR / "annotation" / "_annotations.coco.json"),
    )
)
IMAGES_DIR: Path = Path(
    os.environ.get("VISIONQC_IMAGES_DIR", str(DATASET_DIR / "images"))
)

# ---------------------------------------------------------------------------
# Prepared YOLO-format dataset
# ---------------------------------------------------------------------------
PREPARED_DIR: Path = Path(
    os.environ.get(
        "VISIONQC_PREPARED_DIR",
        str(PROJECT_ROOT / "data" / "prepared"),
    )
)
DATASET_YAML: Path = PREPARED_DIR / "dataset.yaml"

# ---------------------------------------------------------------------------
# Model artifacts
# ---------------------------------------------------------------------------
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts" / "models" / "pcb_yolov8n"

# Experiment A (640px) — preferred if trained
_EXP_A_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "pcb_yolov8n_640" / "weights" / "best.pt"
_BASELINE_MODEL_PATH = ARTIFACTS_DIR / "weights" / "best.pt"

# Auto-select best available model (env var → exp_a → baseline)
_DEFAULT_MODEL_PATH = str(
    _EXP_A_MODEL_PATH if _EXP_A_MODEL_PATH.exists() else _BASELINE_MODEL_PATH
)
MODEL_PATH: Path = Path(
    os.environ.get("VISIONQC_MODEL_PATH", _DEFAULT_MODEL_PATH)
)

# ---------------------------------------------------------------------------
# Inference settings
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("VISIONQC_CONFIDENCE_THRESHOLD", "0.25")
)
IOU_THRESHOLD: float = float(
    os.environ.get("VISIONQC_IOU_THRESHOLD", "0.45")
)

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
APP_TITLE = "VisionQC"
APP_DESCRIPTION = "Automated PCB Defect Inspection API"
APP_VERSION = "0.3.0"
ACCEPTED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png"}

# ---------------------------------------------------------------------------
# Defect classes
# ---------------------------------------------------------------------------
# Original COCO category IDs → class name (category 0 is root label, excluded)
DEFECT_CLASSES = {
    1: "missing_pad",
    2: "mouse_bite",
    3: "open_circuit",
    4: "short",
    5: "spur",
    6: "spurious_copper",
}

# Model class IDs (0-based, contiguous) → class name
# This mapping is produced by dataset_prep.py and must stay consistent.
YOLO_CLASS_NAMES: list[str] = [
    "missing_pad",     # model ID 0
    "mouse_bite",      # model ID 1
    "open_circuit",    # model ID 2
    "short",           # model ID 3
    "spur",            # model ID 4
    "spurious_copper", # model ID 5
]

# BGR colour palette for OpenCV bbox drawing (indexed by YOLO class ID 0-5)
DEFECT_COLOURS_BY_MODEL_ID: dict[int, tuple[int, int, int]] = {
    0: (255, 100, 100),   # missing_pad
    1: (100, 255, 100),   # mouse_bite
    2: (100, 100, 255),   # open_circuit
    3: (255, 220,  50),   # short
    4: (255, 100, 255),   # spur
    5: ( 50, 220, 255),   # spurious_copper
}

# Legacy mapping for GT visualizer (COCOAnnotationPredictor, Phase 1 COCO IDs)
DEFECT_COLOURS = {
    1: (255, 100, 100),
    2: (100, 255, 100),
    3: (100, 100, 255),
    4: (255, 220,  50),
    5: (255, 100, 255),
    6: ( 50, 220, 255),
}

# ---------------------------------------------------------------------------
# Reports / docs directories
# ---------------------------------------------------------------------------
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
