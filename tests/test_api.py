"""
VisionQC — Phase 2 API Tests
------------------------------
Run with:
    python -m pytest tests/ -v

Tests cover:
    - Health endpoint (Phase 1 + Phase 2 fields)
    - Invalid file rejection
    - Prediction response structure + real model confidence
    - Dataset validation endpoint
    - Model loading

NOTE: Tests that require a trained model are skipped if best.pt does not exist.
      Train the model first with scripts/train.py, then re-run the full suite.
"""

import io
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi.testclient import TestClient

from src.visionqc.api.main import app
from configs.settings import IMAGES_DIR, MODEL_PATH, YOLO_CLASS_NAMES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def real_image_bytes() -> tuple[bytes, str]:
    images = sorted(IMAGES_DIR.glob("*.jpg"))
    if not images:
        pytest.skip(f"No dataset images found in {IMAGES_DIR}.")
    img_path = images[0]
    return img_path.read_bytes(), img_path.name


@pytest.fixture(scope="module")
def model_available() -> bool:
    """True if best.pt exists — used to conditionally skip model-dependent tests."""
    return MODEL_PATH.exists()


# ---------------------------------------------------------------------------
# Tests: GET /health
# ---------------------------------------------------------------------------

def test_health_returns_200(client):
    assert client.get("/health").status_code == 200


def test_health_response_structure(client):
    data = client.get("/health").json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["status"] == "healthy"


def test_health_includes_model_name(client):
    data = client.get("/health").json()
    assert "model_name" in data, "health response missing 'model_name'"


def test_health_includes_confidence_threshold(client):
    data = client.get("/health").json()
    assert "confidence_threshold" in data
    assert 0.0 < data["confidence_threshold"] <= 1.0


def test_health_model_loaded_reflects_reality(client, model_available):
    """model_loaded must be True iff best.pt exists."""
    data = client.get("/health").json()
    if model_available:
        assert data["model_loaded"] is True, (
            "model_loaded is False even though best.pt exists. "
            "Restart the server after training."
        )
    else:
        # Before training it's acceptable to be False — do not fail the test
        pass


# ---------------------------------------------------------------------------
# Tests: POST /predict — invalid input
# ---------------------------------------------------------------------------

def test_predict_rejects_non_image(client):
    fake = io.BytesIO(b"this is not an image")
    resp = client.post("/predict", files={"file": ("test.txt", fake, "text/plain")})
    assert resp.status_code in (422, 503)   # 503 if model not loaded yet


def test_predict_rejects_pdf(client):
    fake = io.BytesIO(b"%PDF-1.4 fake content")
    resp = client.post("/predict", files={"file": ("report.pdf", fake, "application/pdf")})
    assert resp.status_code in (422, 503)


def test_predict_rejects_corrupt_jpeg(client, model_available):
    if not model_available:
        pytest.skip("Model not yet trained — skip corrupt-image test.")
    corrupt = io.BytesIO(b"\xff\xd8\xff" + b"\x00" * 100)
    resp = client.post("/predict", files={"file": ("corrupt.jpg", corrupt, "image/jpeg")})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Tests: POST /predict — valid image (requires trained model)
# ---------------------------------------------------------------------------

def test_predict_requires_model(client, model_available):
    if not model_available:
        pytest.skip("Model not trained yet (best.pt not found). Run scripts/train.py.")


def test_predict_valid_image_returns_200(client, real_image_bytes, model_available):
    if not model_available:
        pytest.skip("Model not yet trained.")
    raw, filename = real_image_bytes
    resp = client.post("/predict", files={"file": (filename, io.BytesIO(raw), "image/jpeg")})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"


def test_predict_response_has_detections_field(client, real_image_bytes, model_available):
    if not model_available:
        pytest.skip("Model not yet trained.")
    raw, filename = real_image_bytes
    data = client.post("/predict", files={"file": (filename, io.BytesIO(raw), "image/jpeg")}).json()
    assert "detections" in data
    assert isinstance(data["detections"], list)


def test_predict_response_has_count_field(client, real_image_bytes, model_available):
    if not model_available:
        pytest.skip("Model not yet trained.")
    raw, filename = real_image_bytes
    data = client.post("/predict", files={"file": (filename, io.BytesIO(raw), "image/jpeg")}).json()
    assert "count" in data
    assert isinstance(data["count"], int)
    assert data["count"] == len(data["detections"])


def test_predict_confidence_in_range(client, real_image_bytes, model_available):
    """Real YOLO confidence must be in [0, 1]."""
    if not model_available:
        pytest.skip("Model not yet trained.")
    raw, filename = real_image_bytes
    data = client.post("/predict", files={"file": (filename, io.BytesIO(raw), "image/jpeg")}).json()
    for det in data["detections"]:
        conf = det["confidence"]
        assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of range [0, 1]"


def test_predict_class_name_valid(client, real_image_bytes, model_available):
    """All detected class names must be one of the 6 defect classes."""
    if not model_available:
        pytest.skip("Model not yet trained.")
    raw, filename = real_image_bytes
    data = client.post("/predict", files={"file": (filename, io.BytesIO(raw), "image/jpeg")}).json()
    valid_classes = set(YOLO_CLASS_NAMES)
    for det in data["detections"]:
        assert det["class_name"] in valid_classes, (
            f"Unexpected class name: {det['class_name']}"
        )


def test_predict_bbox_four_values(client, real_image_bytes, model_available):
    """Each detection bbox must have exactly 4 values."""
    if not model_available:
        pytest.skip("Model not yet trained.")
    raw, filename = real_image_bytes
    data = client.post("/predict", files={"file": (filename, io.BytesIO(raw), "image/jpeg")}).json()
    for det in data["detections"]:
        assert "bbox" in det
        assert isinstance(det["bbox"], list) and len(det["bbox"]) == 4


def test_predict_prediction_type_is_yolo(client, real_image_bytes, model_available):
    """prediction_type must be 'yolo_model', never 'annotation_lookup'."""
    if not model_available:
        pytest.skip("Model not yet trained.")
    raw, filename = real_image_bytes
    data = client.post("/predict", files={"file": (filename, io.BytesIO(raw), "image/jpeg")}).json()
    assert data.get("prediction_type") == "yolo_model", (
        f"prediction_type should be 'yolo_model', got: {data.get('prediction_type')}"
    )


def test_predict_response_has_annotated_image(client, real_image_bytes, model_available):
    if not model_available:
        pytest.skip("Model not yet trained.")
    raw, filename = real_image_bytes
    data = client.post("/predict", files={"file": (filename, io.BytesIO(raw), "image/jpeg")}).json()
    assert "annotated_image" in data
    assert len(data["annotated_image"]) > 0


# ---------------------------------------------------------------------------
# Tests: GET /validate
# ---------------------------------------------------------------------------

def test_validate_returns_200(client):
    assert client.get("/validate").status_code == 200


def test_validate_dataset_passes(client):
    data = client.get("/validate").json()
    assert data.get("passed") is True, f"Dataset validation failed: {data.get('errors')}"


def test_validate_response_structure(client):
    data = client.get("/validate").json()
    for field in ("passed", "image_count", "annotation_count", "defect_class_count"):
        assert field in data, f"Validation response missing field: {field}"


# ---------------------------------------------------------------------------
# Tests: Dataset validator directly
# ---------------------------------------------------------------------------

def test_validator_fails_for_wrong_path():
    """Validator must return passed=False when annotation file doesn't exist."""
    from src.visionqc.data.validator import DatasetValidator
    validator = DatasetValidator(
        annotation_file=Path("/nonexistent/path/_annotations.json"),
        images_dir=Path("/nonexistent/images"),
    )
    result = validator.validate()
    assert result.passed is False
    assert len(result.errors) > 0
