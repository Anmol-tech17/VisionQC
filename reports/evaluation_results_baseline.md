# VisionQC — Evaluation: baseline

**Model:** `C:\Users\dell\OneDrive\Desktop\MLOPS CP\VisionQC\artifacts\models\pcb_yolov8n_640\weights\best.pt`
**Split:** test

## Overall Metrics

| Metric | Value |
|---|---|
| Precision | 0.8192 |
| Recall | 0.6954 |
| mAP@0.5 | 0.7527 |
| mAP@0.5:0.95 | 0.3610 |

## Per-Class Metrics

| Class | Precision | Recall | AP@0.5 |
|---|---|---|---|
| missing_pad | 0.8386 | 0.5455 | 0.6697 |
| mouse_bite | 0.6444 | 0.4259 | 0.4470 |
| open_circuit | 0.7996 | 0.7639 | 0.8004 |
| short | 0.8846 | 0.8537 | 0.8896 |
| spur | 0.9899 | 0.6667 | 0.7890 |
| spurious_copper | 0.7579 | 0.9167 | 0.9207 |

## Inference Latency (CPU)

| Metric | Value |
|---|---|
| Samples | 20 |
| Mean | 164.9 ms |
| Min | 115.8 ms |
| Max | 305.8 ms |
| P95 | 220.4 ms |

> All values from actual inference on held-out test split.