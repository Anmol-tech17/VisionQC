"""CI-safe tests for VisionQC — no model weights or MLflow required."""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_imports():
    """Verify core modules import without error."""
    from src.visionqc.inference.predictor import Detection
    from src.visionqc.inference.visualizer import draw_detections
    from configs.settings import YOLO_CLASS_NAMES, DEFECT_CLASSES
    assert len(YOLO_CLASS_NAMES) >= 6
    assert len(DEFECT_CLASSES) >= 6


def test_detection_dataclass():
    """Test Detection dataclass creation and serialization."""
    from src.visionqc.inference.predictor import Detection
    d = Detection(
        class_name="missing_pad",
        class_id=0,
        confidence=0.95,
        bbox=[10.0, 20.0, 30.0, 40.0],
        source="yolo_model"
    )
    assert d.class_name == "missing_pad"
    assert d.confidence == 0.95
    d_dict = d.to_dict()
    assert d_dict["class_id"] == 0
    assert isinstance(d_dict["bbox"], list)


def test_api_health_no_model():
    """Test API health endpoint when model is not loaded (CI environment)."""
    # Mock MLflow and YOLO so the app can start without weights
    with patch("src.visionqc.api.main.model", None), \
         patch("src.visionqc.api.main.model_version", "test"), \
         patch("src.visionqc.api.main.model_name", "test-model"):
        from fastapi.testclient import TestClient
        from src.visionqc.api.main import app
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/health")
        assert response.status_code in (200, 503)
        data = response.json()
        assert "status" in data


def test_api_predict_no_model():
    """Test that /predict returns 503 when model is unavailable."""
    with patch("src.visionqc.api.main.model", None), \
         patch("src.visionqc.api.main.model_version", "test"):
        from fastapi.testclient import TestClient
        from src.visionqc.api.main import app
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", b"\xff\xd8\xff\xe0fake", "image/jpeg")}
        )
        assert response.status_code == 503


def test_settings_config():
    """Verify settings module loads and has expected attributes."""
    from configs.settings import (
        APP_TITLE, CONFIDENCE_THRESHOLD, IOU_THRESHOLD,
        YOLO_CLASS_NAMES, DEFECT_COLOURS_BY_MODEL_ID,
    )
    assert APP_TITLE == "VisionQC"
    assert 0 < CONFIDENCE_THRESHOLD < 1
    assert 0 < IOU_THRESHOLD < 1
    assert len(DEFECT_COLOURS_BY_MODEL_ID) >= 6
