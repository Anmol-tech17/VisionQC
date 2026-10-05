"""
VisionQC — Bounding Box Visualizer
-------------------------------------
Draws detections onto an image and returns annotated JPEG bytes.

Usage
-----
    from src.visionqc.inference.visualizer import draw_detections

    annotated_bytes = draw_detections(raw_image_bytes, detections)
"""

import io
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from configs.settings import DEFECT_COLOURS, DEFECT_COLOURS_BY_MODEL_ID
from src.visionqc.inference.predictor import Detection

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Drawing constants
# ---------------------------------------------------------------------------
BOX_THICKNESS = 3          # pixels
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2
LABEL_PAD = 6              # pixels of padding inside the label background
MAX_DISPLAY_DIM = 1200     # resize long edge to this for display; 0 = no resize


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def draw_detections(
    image_bytes: bytes,
    detections: list[Detection],
    max_dim: Optional[int] = MAX_DISPLAY_DIM,
) -> bytes:
    """
    Draw bounding boxes and labels on a copy of the image.

    Parameters
    ----------
    image_bytes : bytes
        Raw bytes of the uploaded image (JPEG or PNG).
    detections : list[Detection]
        Detections to draw.
    max_dim : int, optional
        If given, scale the image so its longest edge is at most this many
        pixels.  Bounding boxes are scaled accordingly.

    Returns
    -------
    bytes
        JPEG bytes of the annotated image.
    """
    # --- Decode image -------------------------------------------------------
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = pil_img.size

    # Optionally downscale for display
    scale = 1.0
    if max_dim and max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

    # Convert to OpenCV BGR
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # --- Draw each detection ------------------------------------------------
    for det in detections:
        x, y, w, h = det.bbox
        x1 = int(x * scale)
        y1 = int(y * scale)
        x2 = int((x + w) * scale)
        y2 = int((y + h) * scale)

        # Pick colour by class_id (or class_name fallback)
        colour = _class_colour(det)

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, BOX_THICKNESS)

        # Label background + text
        label = f"{det.class_name} {det.confidence:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, FONT, FONT_SCALE, FONT_THICKNESS
        )
        label_y1 = max(y1 - text_h - LABEL_PAD * 2, 0)
        label_y2 = label_y1 + text_h + LABEL_PAD * 2
        label_x2 = x1 + text_w + LABEL_PAD * 2

        cv2.rectangle(img, (x1, label_y1), (label_x2, label_y2), colour, -1)
        cv2.putText(
            img,
            label,
            (x1 + LABEL_PAD, label_y2 - LABEL_PAD),
            FONT,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    # --- Encode back to JPEG ------------------------------------------------
    success, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise RuntimeError("Failed to encode annotated image as JPEG")

    return buf.tobytes()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _class_colour(det: "Detection") -> tuple[int, int, int]:
    """Return the BGR colour for a detection, using model class ID when available."""
    # Use model class_id (0-based) first — works for both YOLO and GT detections
    if hasattr(det, "class_id") and det.class_id is not None:
        return DEFECT_COLOURS_BY_MODEL_ID.get(det.class_id, (200, 200, 200))
    # Fallback: look up by class name (legacy COCO IDs 1-6)
    class_to_coco_id = {
        "missing_pad": 1, "mouse_bite": 2, "open_circuit": 3,
        "short": 4, "spur": 5, "spurious_copper": 6,
    }
    cat_id = class_to_coco_id.get(det.class_name, 1)
    return DEFECT_COLOURS.get(cat_id, (200, 200, 200))

