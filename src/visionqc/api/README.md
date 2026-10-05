# VisionQC API

This directory contains the FastAPI inference service for the VisionQC project. It serves predictions using the latest "champion" model registered in MLflow.

## Setup & Running

1. **Activate Environment:**
   Ensure your Python virtual environment is activated and dependencies (`fastapi`, `uvicorn`, `python-multipart`, `Pillow`) are installed.

2. **Start the API:**
   From the project root (`VisionQC/`), run:
   ```bash
   uvicorn src.visionqc.api.main:app --reload
   ```
   The API will start on `http://127.0.0.1:8000`.

## Endpoints

### `GET /health`
Returns the status of the API and the version of the loaded champion model.

**Response (200 OK):**
```json
{
  "status": "ok",
  "model_version": 2
}
```

### `POST /predict`
Uploads an image to receive defect predictions.

**Request:**
- **Method:** POST
- **Content-Type:** `multipart/form-data`
- **Body:** Form field `file` containing the image file.

**Example using `curl`:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@data/prepared/test/images/pcb_defect_007.jpg"
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
    },
    // ... more detections
  ]
}
```

## How It Works
- On startup, the API queries the MLflow registry for `models:/VisionQC-Detector@champion`.
- It dynamically downloads the weights for that champion and loads them using Ultralytics `YOLO`.
- When an image is submitted, it is converted to RGB format and run through inference.
- The raw YOLO output is serialized into JSON using `results.to_json()` and returned to the client.
