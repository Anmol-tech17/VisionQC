# VisionQC — Evaluation: pcb_yolov8n_exp_b

**Model:** `artifacts\models\pcb_yolov8n\weights\best.pt`
**Split:** test

## Overall Metrics

| Metric | Value |
|---|---|
| Precision | 0.4478 |
| Recall | 0.0359 |
| mAP@0.5 | 0.0337 |
| mAP@0.5:0.95 | 0.0083 |

## Per-Class Metrics

| Class | Precision | Recall | AP@0.5 |
|---|---|---|---|
| missing_pad | 0.0253 | 0.0227 | 0.0205 |
| mouse_bite | 1.0000 | 0.0000 | 0.0006 |
| open_circuit | 0.2921 | 0.0638 | 0.0720 |
| short | 0.2151 | 0.0732 | 0.0531 |
| spur | 1.0000 | 0.0000 | 0.0038 |
| spurious_copper | 0.1541 | 0.0557 | 0.0523 |
| missing_hole | N/A | N/A | N/A |

## Inference Latency (CPU)

| Metric | Value |
|---|---|
| Samples | 20 |
| Mean | 103.2 ms |
| Min | 64.2 ms |
| Max | 246.6 ms |
| P95 | 194.6 ms |

> All values from actual inference on held-out test split.