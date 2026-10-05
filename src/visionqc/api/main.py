"""
VisionQC — FastAPI Application (Phase 2)
------------------------------------------
Uses the trained YOLOv8n model for REAL inference.
The COCOAnnotationPredictor (Phase 1) is NOT used for predictions.

Endpoints
---------
GET  /              → serves index.html
GET  /health        → returns status, model info, confidence threshold
POST /predict       → real YOLO model inference → detections + annotated image
GET  /validate      → runs dataset validation
"""

import base64
import io
import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Resolve project root
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.settings import (
    ACCEPTED_IMAGE_TYPES,
    APP_DESCRIPTION,
    APP_TITLE,
    APP_VERSION,
    CONFIDENCE_THRESHOLD,
    MODEL_PATH,
)
from src.visionqc.data.validator import DatasetValidator
from src.visionqc.inference.predictor import YOLOPredictor
from src.visionqc.inference.visualizer import draw_detections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
predictor = YOLOPredictor()
_dataset_valid: bool = False
_startup_time: float = 0.0


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _dataset_valid, _startup_time
    _startup_time = time.time()
    logger.info("VisionQC %s starting up …", APP_VERSION)

    # Validate dataset
    validator = DatasetValidator()
    result = validator.validate()
    _dataset_valid = result.passed
    logger.info(result.summary_text())

    # Load YOLO model
    success = predictor.load()
    if success:
        logger.info("YOLO model loaded: %s", predictor.model_name)
    else:
        logger.warning(
            "Model not loaded. /predict will return 503 until model is available. "
            "Run scripts/train.py to train the model."
        )

    yield
    logger.info("VisionQC shutting down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan,
)

_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/health")
async def health() -> JSONResponse:
    """
    Application health check.

    Returns model status, dataset validity, confidence threshold, and uptime.
    """
    uptime = round(time.time() - _startup_time, 1) if _startup_time else 0
    return JSONResponse({
        "status":               "healthy",
        "model_loaded":         predictor.is_loaded,
        "model_name":           predictor.model_name,
        "model_path":           str(MODEL_PATH),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "dataset_valid":        _dataset_valid,
        "uptime_seconds":       uptime,
        "version":              APP_VERSION,
    })


@app.get("/validate")
async def validate_dataset() -> JSONResponse:
    """Run the dataset validator and return detailed results."""
    validator = DatasetValidator()
    result = validator.validate()
    return JSONResponse(result.to_dict())


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Run real PCB defect inspection using the trained YOLOv8n model.

    Parameters
    ----------
    file : UploadFile
        PCB image (JPEG or PNG).

    Returns
    -------
    JSON with:
        detections      — list of {class_name, class_id, confidence, bbox, source}
        count           — number of detections
        annotated_image — base64 JPEG with model bounding boxes
        filename        — original filename
        prediction_type — "yolo_model" (never "annotation_lookup")
        confidence_threshold — threshold used for this prediction
    """
    # 1. Check model is available
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. The trained model file was not found. "
                "Run scripts/train.py to train the model, then restart the server."
            ),
        )

    # 2. Validate content type
    content_type = (file.content_type or "").lower()
    if content_type not in ACCEPTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file type: '{file.content_type}'. "
                f"Accepted: {sorted(ACCEPTED_IMAGE_TYPES)}"
            ),
        )

    # 3. Read bytes
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    # 4. Validate it's a real image
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="The uploaded file could not be decoded as an image.",
        )

    # 5. Run YOLO inference
    filename = file.filename or "unknown.jpg"
    try:
        detections = predictor.predict(image_bytes)
    except Exception as exc:
        logger.error("Inference error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}",
        )

    # 6. Draw bounding boxes (using model predictions)
    try:
        annotated_bytes = draw_detections(image_bytes, detections)
        annotated_b64 = base64.b64encode(annotated_bytes).decode("utf-8")
    except Exception as exc:
        logger.error("Visualization failed: %s", exc)
        annotated_b64 = ""

    # 7. Build response
    return JSONResponse({
        "detections":            [d.to_dict() for d in detections],
        "count":                 len(detections),
        "annotated_image":       annotated_b64,
        "filename":              filename,
        "prediction_type":       "yolo_model",
        "confidence_threshold":  CONFIDENCE_THRESHOLD,
        "model_name":            predictor.model_name,
    })
