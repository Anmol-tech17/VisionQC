"""
VisionQC — FastAPI Inference Service
======================================
Serves the PCB defect detection API and monitoring UI.

Endpoints:
    GET  /            → Web UI (static index.html)
    GET  /health      → Health check + model info
    POST /predict     → PCB image defect detection
    GET  /metrics     → Prometheus metrics
    GET  /docs        → Swagger/OpenAPI docs
"""

import base64
import io
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics (lightweight: prometheus_client)
# ---------------------------------------------------------------------------
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry, REGISTRY,
    )
    from starlette.responses import Response as StarletteResponse

    REQUEST_COUNT = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"],
    )
    REQUEST_DURATION = Histogram(
        "http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "endpoint"],
        buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    REQUESTS_IN_PROGRESS = Gauge(
        "http_requests_in_progress",
        "Number of HTTP requests currently being processed",
        ["method", "endpoint"],
    )
    PREDICTION_COUNT = Counter(
        "visionqc_predictions_total",
        "Total prediction requests",
        ["status"],  # "success", "error", "no_model"
    )
    INFERENCE_DURATION = Histogram(
        "visionqc_inference_duration_seconds",
        "Model inference duration in seconds",
        buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    )
    PROMETHEUS_AVAILABLE = True
    logger.info("Prometheus metrics enabled.")
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed — /metrics endpoint disabled.")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_STATIC_DIR = _THIS_DIR / "static"

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
model = None
model_version = None
model_name = "VisionQC-Detector"

CLASS_NAMES = [
    "missing_pad", "mouse_bite", "open_circuit",
    "short", "spur", "spurious_copper",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_version, model_name
    logger.info("Initializing FastAPI service...")

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient()

        champion_meta = client.get_model_version_by_alias("VisionQC-Detector", "champion")
        model_version = champion_meta.version
        logger.info(f"Loading VisionQC-Detector@champion (v{model_version})...")

        weights_dir = mlflow.artifacts.download_artifacts(
            artifact_uri="models:/VisionQC-Detector@champion"
        )
        weights_path = os.path.join(weights_dir, "artifacts", "best.pt")

        from ultralytics import YOLO
        model = YOLO(weights_path)
        model_name = f"VisionQC-Detector v{model_version}"
        logger.info("Champion model loaded successfully via MLflow.")
    except Exception as e:
        logger.warning(f"MLflow model load failed: {e}")
        logger.info("Attempting fallback model load from settings...")
        try:
            from configs.settings import MODEL_PATH
            from ultralytics import YOLO
            if Path(str(MODEL_PATH)).exists():
                model = YOLO(str(MODEL_PATH))
                model_version = "local"
                model_name = f"YOLOv8n-PCB (local: {Path(str(MODEL_PATH)).parent.parent.name})"
                logger.info(f"Fallback model loaded: {MODEL_PATH}")
            else:
                logger.error(f"Fallback model not found at {MODEL_PATH}")
                model = None
                model_version = "unavailable"
        except Exception as e2:
            logger.error(f"Fallback load also failed: {e2}")
            model = None
            model_version = "unavailable"

    yield

    logger.info("Shutting down FastAPI service...")
    model = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VisionQC Inference API",
    description="API for PCB Defect Detection — Automated Manufacturing Quality Control",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Prometheus middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    if not PROMETHEUS_AVAILABLE:
        return await call_next(request)

    # Skip metrics endpoint itself
    if request.url.path == "/metrics":
        return await call_next(request)

    method = request.method
    # Normalize path to avoid high-cardinality labels
    path = request.url.path
    if path.startswith("/static"):
        path = "/static"

    REQUESTS_IN_PROGRESS.labels(method=method, endpoint=path).inc()
    start = time.perf_counter()
    try:
        response = await call_next(request)
        duration = time.perf_counter() - start
        REQUEST_COUNT.labels(
            method=method, endpoint=path, status_code=response.status_code
        ).inc()
        REQUEST_DURATION.labels(method=method, endpoint=path).observe(duration)
        return response
    except Exception as exc:
        duration = time.perf_counter() - start
        REQUEST_COUNT.labels(method=method, endpoint=path, status_code=500).inc()
        REQUEST_DURATION.labels(method=method, endpoint=path).observe(duration)
        raise
    finally:
        REQUESTS_IN_PROGRESS.labels(method=method, endpoint=path).dec()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui():
    """Serve the VisionQC web dashboard."""
    index_path = _STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return HTMLResponse("<h1>VisionQC API</h1><p>UI not found. Visit /docs for API documentation.</p>")


@app.get("/health")
async def health_check():
    """Health check endpoint with model and system information."""
    loaded = model is not None
    status_code = 200 if loaded else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if loaded else "degraded",
            "model_loaded": loaded,
            "model_version": model_version,
            "model_name": model_name if loaded else "unavailable",
            "confidence_threshold": 0.25,
            "class_names": CLASS_NAMES,
            "num_classes": len(CLASS_NAMES),
        },
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Run PCB defect detection on an uploaded image.

    Returns detected defects with bounding boxes, class labels,
    confidence scores, and an annotated image.
    """
    if model is None:
        if PROMETHEUS_AVAILABLE:
            PREDICTION_COUNT.labels(status="no_model").inc()
        raise HTTPException(status_code=503, detail="Model is not available")

    if not file.content_type or not file.content_type.startswith("image/"):
        if PROMETHEUS_AVAILABLE:
            PREDICTION_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image (JPEG/PNG).")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        if PROMETHEUS_AVAILABLE:
            PREDICTION_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

    try:
        # Measure inference time
        t0 = time.perf_counter()
        results = model.predict(image, verbose=False, conf=0.25)
        inference_time = time.perf_counter() - t0

        if PROMETHEUS_AVAILABLE:
            INFERENCE_DURATION.observe(inference_time)

        # Parse detections
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf_val = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                detections.append({
                    "class_name": class_name,
                    "class_id": cls_id,
                    "confidence": round(conf_val, 4),
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                })

        detections.sort(key=lambda d: d["confidence"], reverse=True)

        # Generate annotated image
        annotated_b64 = None
        try:
            import cv2
            import numpy as np

            # Class colors (BGR for OpenCV)
            CLASS_COLORS_BGR = [
                (100, 100, 255),  # missing_pad - red
                (100, 255, 100),  # mouse_bite - green
                (255, 100, 100),  # open_circuit - blue
                (50, 220, 255),   # short - gold
                (255, 100, 255),  # spur - magenta
                (255, 220, 50),   # spurious_copper - cyan
            ]

            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            for det in detections:
                x, y, w, h = det["bbox"]
                x1_d, y1_d = int(x), int(y)
                x2_d, y2_d = int(x + w), int(y + h)
                color = CLASS_COLORS_BGR[det["class_id"]] if det["class_id"] < len(CLASS_COLORS_BGR) else (200, 200, 200)
                cv2.rectangle(img_bgr, (x1_d, y1_d), (x2_d, y2_d), color, 3)

                label = f"{det['class_name']} {det['confidence']:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(img_bgr, (x1_d, max(y1_d - th - 12, 0)), (x1_d + tw + 12, max(y1_d, th + 12)), color, -1)
                cv2.putText(img_bgr, label, (x1_d + 6, max(y1_d - 6, th + 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            annotated_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        except Exception as viz_err:
            logger.warning(f"Annotation visualization failed: {viz_err}")

        if PROMETHEUS_AVAILABLE:
            PREDICTION_COUNT.labels(status="success").inc()

        return {
            "predictions": detections,
            "detections": detections,
            "count": len(detections),
            "annotated_image": annotated_b64,
            "prediction_type": "yolo_model",
            "inference_time_ms": round(inference_time * 1000, 1),
            "model_name": model_name,
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        if PROMETHEUS_AVAILABLE:
            PREDICTION_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint."""
    if not PROMETHEUS_AVAILABLE:
        return JSONResponse(
            status_code=501,
            content={"detail": "prometheus_client not installed"},
        )
    from starlette.responses import Response
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )
