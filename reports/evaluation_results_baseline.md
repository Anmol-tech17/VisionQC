# VisionQC — Evaluation: baseline

**Model:** `artifacts\models\pcb_yolov8n_baseline\weights\best.pt`
**Split:** test

## Overall Metrics

| Metric | Value |
|---|---|
| Precision | 0.2470 |
| Recall | 0.1668 |
| mAP@0.5 | 0.1650 |
| mAP@0.5:0.95 | 0.0670 |

## Per-Class Metrics

| Class | Precision | Recall | AP@0.5 |
|---|---|---|---|
| missing_pad | 0.1123 | 0.0227 | 0.0510 |
| mouse_bite | 0.0000 | 0.0000 | 0.0141 |
| open_circuit | 0.4386 | 0.2553 | 0.2422 |
| short | 0.5243 | 0.1951 | 0.1987 |
| spur | 0.0000 | 0.0000 | 0.0221 |
| spurious_copper | 0.4066 | 0.5278 | 0.4621 |

## Inference Latency (CPU)

| Metric | Value |
|---|---|
| Samples | 20 |
| Mean | 251.2 ms |
| Min | 121.2 ms |
| Max | 786.3 ms |
| P95 | 573.7 ms |

> All values from actual inference on held-out test split.