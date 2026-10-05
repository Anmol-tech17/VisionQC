"""
VisionQC — Model Loader (Singleton)
--------------------------------------
Loads the trained YOLOv8n model exactly once at application startup.
Thread-safe via a threading.Lock.

Usage
-----
    from src.visionqc.inference.model_loader import get_model_loader

    loader = get_model_loader()
    loader.load()
    model = loader.model
"""

import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_INSTANCE: Optional["ModelLoader"] = None


class ModelLoader:
    """
    Singleton wrapper around an ultralytics YOLO model.

    Parameters
    ----------
    model_path : Path
        Path to the trained best.pt checkpoint.
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._model = None
        self._loaded: bool = False
        self._load_error: Optional[str] = None
        self._model_name: str = "yolov8n-pcb"
        self._num_classes: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """
        Load the model. Safe to call multiple times (no-op if already loaded).

        Returns
        -------
        bool
            True if model is loaded successfully, False otherwise.
        """
        with _LOCK:
            if self._loaded:
                return True

            if not self.model_path.exists():
                self._load_error = (
                    f"Model file not found: {self.model_path}\n"
                    "Run scripts/train.py to train the model first."
                )
                logger.error(self._load_error)
                return False

            try:
                from ultralytics import YOLO
                logger.info("Loading YOLO model from: %s", self.model_path)
                self._model = YOLO(str(self.model_path))

                # Extract metadata
                if hasattr(self._model, "model") and hasattr(self._model.model, "nc"):
                    self._num_classes = self._model.model.nc
                elif hasattr(self._model, "overrides"):
                    self._num_classes = self._model.overrides.get("nc", 6)
                else:
                    self._num_classes = 6   # fallback for PCB dataset

                self._model_name = self.model_path.parent.parent.name or "pcb_yolov8n"
                self._loaded = True
                logger.info(
                    "Model loaded successfully. Classes: %d, Path: %s",
                    self._num_classes, self.model_path,
                )
                return True

            except Exception as exc:
                self._load_error = str(exc)
                logger.error("Failed to load model: %s", exc)
                return False

    @property
    def model(self):
        """The underlying ultralytics YOLO object, or None if not loaded."""
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def model_path_str(self) -> str:
        return str(self.model_path)


def get_model_loader() -> ModelLoader:
    """
    Return the application-wide ModelLoader singleton.
    Creates it on first call using the path from settings.
    """
    global _INSTANCE
    if _INSTANCE is None:
        with _LOCK:
            if _INSTANCE is None:
                from configs.settings import MODEL_PATH
                _INSTANCE = ModelLoader(MODEL_PATH)
    return _INSTANCE
