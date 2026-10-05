"""
VisionQC — Predictors (Phase 2)
---------------------------------
Two predictor classes:

1.  YOLOPredictor   — PRODUCTION predictor. Uses the trained YOLOv8n model
                      to perform real inference. Confidence scores come from
                      the model output.  This is used by the /predict endpoint.

2.  GTAnnotationLookup — GROUND TRUTH helper.  Retained from Phase 1 for
                         ground-truth visualization only.  NOT used by /predict.
                         Clearly labelled as "ground_truth" in API responses.

IMPORTANT:
    YOLOPredictor is the ONLY source of predictions in the production API.
    GTAnnotationLookup must never be called "prediction" or "confidence".
"""

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared Detection dataclass
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """
    A single detected object.

    Attributes
    ----------
    class_name   : Human-readable defect class name
    class_id     : 0-based model class ID
    confidence   : Model confidence score in [0, 1].
                   For GT lookups, this is 1.0 (ground truth, not a score).
    bbox         : Bounding box as [x, y, width, height] in pixel coordinates
    source       : "yolo_model" or "ground_truth"
    """
    class_name: str
    class_id: int
    confidence: float
    bbox: list[float]
    source: str = "yolo_model"

    def to_dict(self) -> dict:
        return {
            "class_name":  self.class_name,
            "class_id":    self.class_id,
            "confidence":  round(self.confidence, 4),
            "bbox":        [round(v, 2) for v in self.bbox],
            "source":      self.source,
        }


# ---------------------------------------------------------------------------
# YOLOPredictor  (PRODUCTION — real ML inference)
# ---------------------------------------------------------------------------

class YOLOPredictor:
    """
    Runs real inference using the trained YOLOv8n model.

    Confidence scores are the raw model output probabilities.
    Bounding boxes come from model predictions, not annotations.

    Parameters
    ----------
    conf_threshold : float
        Minimum confidence to include a detection.
    iou_threshold  : float
        NMS IoU threshold.
    """

    def __init__(
        self,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> None:
        from configs.settings import CONFIDENCE_THRESHOLD, IOU_THRESHOLD, YOLO_CLASS_NAMES
        self._conf = conf_threshold if conf_threshold is not None else CONFIDENCE_THRESHOLD
        self._iou  = iou_threshold  if iou_threshold  is not None else IOU_THRESHOLD
        self._class_names = YOLO_CLASS_NAMES
        self._loader = None

    def load(self) -> bool:
        """Load the model. Returns True on success."""
        from src.visionqc.inference.model_loader import get_model_loader
        self._loader = get_model_loader()
        return self._loader.load()

    def predict(
        self,
        image_bytes: bytes,
        conf_threshold: Optional[float] = None,
    ) -> list[Detection]:
        """
        Run object detection on image bytes.

        Parameters
        ----------
        image_bytes   : Raw JPEG/PNG bytes of the image.
        conf_threshold: Override the default confidence threshold.

        Returns
        -------
        list[Detection]
            Real model detections sorted by descending confidence.
        """
        if self._loader is None or not self._loader.is_loaded:
            logger.error("YOLOPredictor.predict() called before model was loaded.")
            return []

        conf = conf_threshold if conf_threshold is not None else self._conf

        try:
            from PIL import Image
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            results = self._loader.model.predict(
                source=pil_image,
                conf=conf,
                iou=self._iou,
                verbose=False,
                save=False,
            )
        except Exception as exc:
            logger.error("Model inference failed: %s", exc)
            return []

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id   = int(box.cls.item())
                conf_val = float(box.conf.item())
                # xyxy format → xywh
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1

                class_name = (
                    self._class_names[cls_id]
                    if cls_id < len(self._class_names)
                    else f"class_{cls_id}"
                )

                detections.append(Detection(
                    class_name=class_name,
                    class_id=cls_id,
                    confidence=conf_val,
                    bbox=[x1, y1, w, h],
                    source="yolo_model",
                ))

        # Sort by descending confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)
        logger.debug("YOLOPredictor: %d detections (conf≥%.2f)", len(detections), conf)
        return detections

    @property
    def is_loaded(self) -> bool:
        return self._loader is not None and self._loader.is_loaded

    @property
    def model_name(self) -> str:
        return self._loader.model_name if self._loader else "not_loaded"

    @property
    def confidence_threshold(self) -> float:
        return self._conf


# ---------------------------------------------------------------------------
# GTAnnotationLookup  (GROUND TRUTH only — NOT used in /predict)
# ---------------------------------------------------------------------------

class GTAnnotationLookup:
    """
    Looks up COCO ground-truth annotations by filename.

    IMPORTANT: This is NOT a predictor. It returns ground-truth labels, not
    model predictions. Confidence is always 1.0 (not a model score).
    Only use this for ground-truth visualization, NOT for /predict.

    Kept from Phase 1 for potential future GT-vs-prediction comparison views.
    """

    def __init__(self, annotation_file: Optional[Path] = None) -> None:
        from configs.settings import ANNOTATION_FILE, DEFECT_CLASSES
        self.annotation_file = annotation_file or ANNOTATION_FILE
        self._defect_classes = DEFECT_CLASSES
        self._index: dict[str, list[Detection]] = {}
        self._loaded: bool = False

    def load(self) -> None:
        """Load and index the COCO annotations."""
        if self._loaded:
            return
        import json
        with open(self.annotation_file, encoding="utf-8") as f:
            coco = json.load(f)

        id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
        index: dict[str, list[Detection]] = {fn: [] for fn in id_to_filename.values()}

        for ann in coco["annotations"]:
            cat_id = ann.get("category_id", 0)
            if cat_id not in self._defect_classes:
                continue
            img_id = ann["image_id"]
            if img_id not in id_to_filename:
                continue
            filename = id_to_filename[img_id]
            class_name = self._defect_classes[cat_id]
            bbox = ann["bbox"]
            # Map COCO cat_id to 0-based model_id for consistency
            model_id = cat_id - 1  # COCO 1→0, 2→1, ..., 6→5
            index[filename].append(Detection(
                class_name=class_name,
                class_id=model_id,
                confidence=1.0,          # ground truth — not a model score
                bbox=bbox,
                source="ground_truth",   # always label as ground truth
            ))

        self._index = index
        self._loaded = True
        total = sum(len(v) for v in index.values())
        logger.info("GT lookup loaded: %d images, %d annotations.", len(index), total)

    def lookup(self, filename: str) -> list[Detection]:
        """Return ground-truth annotations for the given filename."""
        if not self._loaded:
            self.load()
        bare = Path(filename).name
        return self._index.get(bare, [])

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------------------
# Backward-compatibility alias (Phase 1 tests reference COCOAnnotationPredictor)
# ---------------------------------------------------------------------------
COCOAnnotationPredictor = GTAnnotationLookup
