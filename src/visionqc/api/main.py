import io
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

import mlflow
from mlflow.tracking import MlflowClient
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Global variables for model state
model = None
model_version = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_version
    logger.info("Initializing FastAPI service...")
    
    try:
        client = MlflowClient()
        champion_meta = client.get_model_version_by_alias("VisionQC-Detector", "champion")
        model_version = champion_meta.version
        logger.info(f"Downloading and loading VisionQC-Detector@champion (Version {model_version})...")
        
        # The registered PyFunc model has a bug (`tojson` instead of `to_json`).
        import os
        weights_dir = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/VisionQC-Detector@champion"
        )
        weights_path = os.path.join(weights_dir, "artifacts", "best.pt")
        model = YOLO(weights_path)
        logger.info("Champion model successfully loaded via Ultralytics.")
    except Exception as e:
        logger.error(f"Failed to load champion model: {e}")
        model = None
        model_version = "Unknown"

    yield
    
    logger.info("Shutting down FastAPI service...")
    model = None

app = FastAPI(
    title="VisionQC Inference API",
    description="API for PCB Defect Detection",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    if model is None:
        return JSONResponse(status_code=503, content={"status": "degraded", "model_version": model_version, "detail": "Model not loaded"})
    return {"status": "ok", "model_version": model_version}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available")
        
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
    try:
        results = model.predict(image, verbose=False)
        # Parse the JSON string from the first result
        predictions = json.loads(results[0].to_json())
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


