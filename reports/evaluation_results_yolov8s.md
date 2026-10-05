# VisionQC — Evaluation: yolov8s

**Model:** `artifacts\models\pcb_exp_3\weights\best.pt`
**Split:** test

## Overall Metrics

| Metric | Value |
|---|---|
| Precision | 0.8284 |
| Recall | 0.7863 |
| mAP@0.5 | 0.8137 |
| mAP@0.5:0.95 | 0.4294 |

## Per-Class Metrics

| Class | Precision | Recall | AP@0.5 |
|---|---|---|---|
| missing_pad | 0.6937 | 0.6591 | 0.7230 |
| mouse_bite | 0.6882 | 0.4906 | 0.5045 |
| open_circuit | 0.8908 | 0.8936 | 0.8986 |
| short | 0.9221 | 0.9024 | 0.8985 |
| spur | 0.8575 | 0.8000 | 0.8838 |
| spurious_copper | 0.9179 | 0.9722 | 0.9735 |
| missing_hole | N/A | N/A | N/A |

## Inference Latency (CPU)

| Metric | Value |
|---|---|
| Samples | 20 |
| Mean | 346.0 ms |
| Min | 239.9 ms |
| Max | 509.0 ms |
| P95 | 458.6 ms |

> All values from actual inference on held-out test split.