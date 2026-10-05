# VisionQC — Model Card

## Model Overview

| Item | Value |
|---|---|
| Model name | VisionQC PCB Defect Detector (Exp A — Final) |
| Architecture | YOLOv8n (nano) |
| Task | Object detection (bounding box) |
| Framework | Ultralytics 8.4.160 |
| Base weights | yolov8n.pt (pretrained on COCO) |
| Training type | Transfer learning (fine-tuned on PCB defect dataset) |
| Version | 0.3.0 (Phase 2 Experiment A) |

## Training Configuration

| Parameter | Value |
|---|---|
| Config file | `configs/training_exp_a.yaml` |
| Epochs | 40 / 40 completed |
| Batch size | 4 |
| Image size | **640 × 640 px** |
| Patience | 10 |
| Device | CPU (Intel Core i3-8130U) |
| Workers | 0 |
| Random seed | 42 |
| Confidence threshold | 0.25 |
| IoU threshold | 0.45 |
| Pretrained weights | yolov8n.pt (COCO) |
| Training duration | 86.6 minutes |
| mosaic | OFF (0.0) |
| degrees/shear/perspective | OFF |
| flipLR | 0.5 |
| hsv_s / hsv_v | 0.3 / 0.3 |

**Why 640px?** Dataset audit found 96.3% of bounding boxes < 5% of image dimensions. At 320px median defect = 9×9 pixels (too small); at 640px median defect = 19×19 pixels (reliably detectable). This single change drove the primary improvement.

**Why mosaic OFF?** Mosaic randomly crops and tiles images, destroying the spatial context of tiny PCB defects. Conservative augmentation is appropriate for small, fine-grained defects.

## Dataset

| Item | Value |
|---|---|
| Total images | 230 |
| Train images | 161 |
| Validation images | 34 |
| Test images | 35 |
| Total annotations | 1,704 |
| Classes | 6 |
| Seed | 42 |
| Leakage check | Passed (verified) |

## Defect Classes

| Model ID | Class Name | COCO Category ID |
|---|---|---|
| 0 | missing_pad | 1 |
| 1 | mouse_bite | 2 |
| 2 | open_circuit | 3 |
| 3 | short | 4 |
| 4 | spur | 5 |
| 5 | spurious_copper | 6 |

## Input / Output Format

**Input:** JPEG or PNG image of a PCB board (any resolution; internally resized to 640×640)

**Output (per detection):**
```json
{
  "class_name": "short",
  "class_id": 3,
  "confidence": 0.87,
  "bbox": [x, y, width, height],
  "source": "yolo_model"
}
```

## Validation Metrics (Best Epoch, Val Split)

| Metric | Value |
|---|---|
| Precision | **0.7456** |
| Recall | **0.7004** |
| mAP@0.5 | **0.7400** |
| mAP@0.5:0.95 | **0.3698** |

## Final Test Metrics (Held-out Test Split — Evaluated Once)

| Metric | Value |
|---|---|
| Precision | **0.8192** |
| Recall | **0.6954** |
| mAP@0.5 | **0.7527** |
| mAP@0.5:0.95 | **0.3610** |

## Per-Class Test Metrics

| Class | Precision | Recall | AP@0.5 | AP@0.5:0.95 |
|---|---|---|---|---|
| missing_pad | 0.8390 | 0.5450 | **0.6697** | 0.4450 |
| mouse_bite | 0.6440 | 0.4260 | **0.4470** | 0.1420 |
| open_circuit | 0.8000 | 0.7640 | **0.8004** | 0.2430 |
| short | 0.8850 | 0.8540 | **0.8896** | 0.4400 |
| spur | 0.9900 | 0.6670 | **0.7890** | 0.3070 |
| spurious_copper | 0.7580 | 0.9170 | **0.9207** | 0.5880 |

## Comparison with Baseline (Test Split)

| Metric | Baseline (320px) | **Final (640px)** | Delta |
|---|---|---|---|
| Precision | 0.2470 | **0.8192** | +0.5722 |
| Recall | 0.1668 | **0.6954** | +0.5286 |
| mAP@0.5 | 0.1650 | **0.7527** | +0.5877 |
| mAP@0.5:0.95 | 0.0670 | **0.3610** | +0.2940 |

## Inference Latency (CPU — Intel Core i3-8130U)

| Metric | Value |
|---|---|
| Samples measured | 20 |
| Mean latency | ~185 ms |
| Min latency | ~75 ms |
| Max latency | ~300 ms |
| P95 latency | ~299 ms |

## Quality Gate

Thresholds established at 85% of final test metrics for future MLflow model promotion:

```yaml
quality_gate:
  map50_min: 0.64       # 85% of 0.7527
  precision_min: 0.655  # 85% of 0.8192
  recall_min: 0.556     # 85% of 0.6954
```

## Model Artifacts

```
artifacts/
  models/
    pcb_yolov8n_baseline/     — Baseline model (320px, 20 epochs)
    pcb_yolov8n_640/           — Final model (640px, 40 epochs) ← PROMOTED
      weights/
        best.pt               — Best checkpoint by val mAP50
        last.pt               — Last epoch checkpoint
```

## Reproducibility

```bash
# Reproduce Experiment A from scratch:
C:\Program Files\Python313\python.exe scripts/prepare_dataset.py
C:\Program Files\Python313\python.exe scripts/train.py --config configs/training_exp_a.yaml
C:\Program Files\Python313\python.exe scripts/evaluate.py \
    --model artifacts/models/pcb_yolov8n_640/weights/best.pt --name exp_a
```

Environment:
- Python: 3.13.0
- PyTorch: 2.14.0+cpu
- Ultralytics: 8.4.160
- Seed: 42

## Known Limitations

1. **Small dataset (161 train images):** Generalisation to unseen PCB designs is uncertain
2. **mouse_bite weakest class (AP50=0.447, Recall=0.426):** ~57% of mouse bites missed; these are the physically smallest defects
3. **missing_pad lower recall (0.545):** Absence-of-material defects are harder to detect geometrically
4. **CPU inference only:** ~185ms mean latency limits real-time throughput to ~5 images/second
5. **Single domain:** Dataset comes from one type of PCB; domain shift risk on different boards
6. **No defect-free training images:** False positive rate on normal PCBs is unknown
7. **Not production-ready:** 230 images insufficient for production-grade confidence; needs extensive validation

## Intended Use

- Automated PCB surface defect detection during manufacturing quality inspection
- Research and educational demonstration of object detection in manufacturing (MLOps course)
- Baseline model for MLflow experiment tracking and comparison in later phases

## Out-of-Scope Use

- Safety-critical autonomous inspection without human oversight
- Deployment on PCB types significantly different from the training data
- Any high-stakes decisions without additional validation

## Ethical Considerations

- **False negatives (missed defects):** A defective PCB passed as good could lead to product failures downstream
- **False positives (false alarms):** Good PCBs flagged as defective cause unnecessary rework and cost
- **Automation bias risk:** Operators may over-trust model predictions, especially when recall is 69.5% (30.5% of real defects are missed)
- **Minimum viable for demo purposes only:** Not for actual manufacturing deployment
