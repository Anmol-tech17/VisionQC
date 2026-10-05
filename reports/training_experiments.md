# VisionQC — Training Experiments Report

## Dataset Audit Summary

A full quality audit was run on the prepared YOLO dataset (`scripts/audit_dataset.py`) before any experiments.

**Critical Finding:** 96.3% of all bounding boxes are smaller than 5% of image width or height.

| Split | Images | Annotations | Empty labels | Invalid coords |
|---|---|---|---|---|
| Train | 161 | 1,194 | 0 | 0 |
| Val | 34 | 243 | 0 | 0 |
| Test | 35 | 267 | 0 | 0 |

**Bounding box size analysis (all splits):**

| Metric | Width | Height |
|---|---|---|
| Min | 0.0053 | 0.0052 |
| Mean | 0.0406 | 0.0407 |
| Median | 0.0298 | 0.0303 |
| Max | 0.4898 | 0.4036 |
| **< 5% dim** | **96.3%** | — |

**Effective pixel size of median defect:**

| Image size | Median defect (px) |
|---|---|
| 320 px (baseline) | ~9 × 9 |
| 640 px (Exp A) | ~19 × 19 |

**Conclusion:** At 320px, the vast majority of defects are only 9×9 pixels — far below YOLO's reliable detection threshold. Increasing to 640px was the primary hypothesis for Experiment A.

**Dataset health:** Zero annotation errors, zero invalid coordinates, all 6 classes present in all splits, no image leakage verified.

---

## Class Distribution (All Splits)

| Class | Total | Train | Val | Test | % |
|---|---|---|---|---|---|
| missing_pad | 276 | 193 | 39 | 44 | 16.2% |
| mouse_bite | 356 | 252 | 50 | 54 | 20.9% |
| open_circuit | 276 | 191 | 38 | 47 | 16.2% |
| short | 254 | 177 | 36 | 41 | 14.9% |
| spur | 296 | 208 | 43 | 45 | 17.4% |
| spurious_copper | 246 | 173 | 37 | 36 | 14.4% |

Distribution is relatively balanced (14.4%–20.9%). Class imbalance was NOT the primary cause of baseline weakness.

---

## Experiment Summary

| Experiment | Model | imgsz | Epochs (req.) | Epochs (actual) | Patience | Duration |
|---|---|---|---|---|---|---|
| Baseline | YOLOv8n | 320 px | 20 | 20 | 5 | 16.1 min |
| **Exp A (640px)** | YOLOv8n | **640 px** | 40 | **40** | 10 | **86.6 min** |

---

## Baseline (320px, 20 epochs)

**Configuration:**

| Parameter | Value |
|---|---|
| Model | YOLOv8n (pretrained COCO) |
| Image size | 320 px |
| Epochs | 20 / 20 completed |
| Batch size | 4 |
| Patience | 5 |
| Seed | 42 |
| Augmentation | Ultralytics defaults (including mosaic) |
| Duration | 16.1 minutes |

**Validation metrics (best epoch, val split):**

| Metric | Value |
|---|---|
| Precision | 0.1820 |
| Recall | 0.1806 |
| mAP@0.5 | 0.1604 |
| mAP@0.5:0.95 | 0.0628 |

**Test metrics (held-out test split — evaluated once):**

| Metric | Value |
|---|---|
| Precision | 0.2470 |
| Recall | 0.1668 |
| mAP@0.5 | 0.1650 |
| mAP@0.5:0.95 | 0.0670 |

**Per-class test metrics:**

| Class | Precision | Recall | AP@0.5 |
|---|---|---|---|
| missing_pad | 0.1123 | 0.0227 | 0.0510 |
| mouse_bite | 0.0000 | 0.0000 | 0.0141 |
| open_circuit | 0.4386 | 0.2553 | 0.2422 |
| short | 0.5243 | 0.1951 | 0.1987 |
| spur | 0.0000 | 0.0000 | 0.0221 |
| spurious_copper | 0.4066 | 0.5278 | 0.4621 |

**Inference latency (CPU):** Mean=251ms, P95=574ms

**Root cause of weakness:** At 320px, median defect = 9×9 pixels. `mouse_bite`, `spur`, and `spur` effectively had near-zero detection at this resolution.

---

## Experiment A (640px, PCB-conservative augmentation)

**Rationale:** Address the 96.3% small-object finding. Increase image size so median defect goes from 9×9 to 19×19 pixels. Also disable mosaic (which randomly crops images, destroying tiny defect context) and rotation/shear (PCB orientation is meaningful).

**Configuration:**

| Parameter | Value | Change from Baseline |
|---|---|---|
| Image size | 640 px | **+100%** |
| Epochs | 40 | +20 |
| Patience | 10 | +5 |
| mosaic | 0.0 | **OFF** (was 1.0 default) |
| degrees | 0.0 | OFF (no rotation) |
| shear | 0.0 | OFF |
| perspective | 0.0 | OFF |
| flipud | 0.0 | OFF |
| fliplr | 0.5 | ON (harmless symmetry) |
| hsv_h | 0.0 | OFF (colour meaningful) |
| hsv_s | 0.3 | Mild saturation |
| hsv_v | 0.3 | Mild brightness |
| mixup | 0.0 | OFF |
| Seed | 42 | Same |
| Duration | 86.6 minutes | — |

**Validation metrics (best epoch, val split):**

| Metric | Value |
|---|---|
| Precision | 0.7456 |
| Recall | 0.7004 |
| mAP@0.5 | 0.7400 |
| mAP@0.5:0.95 | 0.3698 |

**Test metrics (held-out test split — evaluated once):**

| Metric | Value |
|---|---|
| Precision | 0.8192 |
| Recall | 0.6954 |
| mAP@0.5 | 0.7527 |
| mAP@0.5:0.95 | 0.3610 |

**Per-class test metrics:**

| Class | Precision | Recall | AP@0.5 | AP@0.5:0.95 |
|---|---|---|---|---|
| missing_pad | 0.8390 | 0.5450 | 0.6697 | 0.4450 |
| mouse_bite | 0.6440 | 0.4260 | 0.4470 | 0.1420 |
| open_circuit | 0.8000 | 0.7640 | 0.8004 | 0.2430 |
| short | 0.8850 | 0.8540 | 0.8896 | 0.4400 |
| spur | 0.9900 | 0.6670 | 0.7890 | 0.3070 |
| spurious_copper | 0.7580 | 0.9170 | 0.9207 | 0.5880 |

**Inference latency (CPU):** Mean=185ms, P95=299ms

---

## Comparison Table (Test Split)

| Experiment | imgsz | Epochs | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Duration |
|---|---|---|---|---|---|---|---|
| Baseline | 320 | 20 | 0.1650 | 0.0670 | 0.2470 | 0.1668 | 16 min |
| **Exp A (final)** | **640** | **40** | **0.7527** | **0.3610** | **0.8192** | **0.6954** | **87 min** |
| **Delta** | — | — | **+0.5877** | **+0.2940** | **+0.5722** | **+0.5286** | — |

---

## Per-Class Improvement (AP@0.5, Test Split)

| Class | Baseline | Exp A | Delta | Explanation |
|---|---|---|---|---|
| missing_pad | 0.0510 | 0.6697 | **+0.619** | Tiny defect; larger imgsz decisive |
| mouse_bite | 0.0141 | 0.4470 | **+0.433** | Smallest defect type; still lowest |
| open_circuit | 0.2422 | 0.8004 | **+0.558** | Medium defect; very large gain |
| short | 0.1987 | 0.8896 | **+0.691** | Medium defect; excellent result |
| spur | 0.0221 | 0.7890 | **+0.767** | Was near-zero; huge recovery |
| spurious_copper | 0.4621 | 0.9207 | **+0.459** | Larger defect; further improved |

---

## Final Model

**Selected model:** Experiment A (640px)  
**Model path:** `artifacts/models/pcb_yolov8n_640/weights/best.pt`  
**Config:** `configs/training_exp_a.yaml`

Selection criterion: Experiment A achieves mAP@0.5 = 0.7527 vs baseline 0.1650 on the test split.

---

## Quality Gate (Established Post-Experiments)

Based on 85% of Experiment A test metrics. Will be enforced by future MLflow model promotion pipeline.

```yaml
quality_gate:
  map50_min: 0.64       # 85% of 0.7527
  precision_min: 0.655  # 85% of 0.8192
  recall_min: 0.556     # 85% of 0.6954
```

These thresholds mean: a new model candidate must achieve at least mAP@0.5 ≥ 0.64 on the validation set before being promoted to the model registry. This is appropriate given the current dataset size — not so strict as to be unachievable, not so loose as to allow regressions.

---

## Analysis of Remaining Weaknesses

`mouse_bite` remains the weakest class (AP50=0.447, recall=0.426). Analysis:
- Mouse bites are physically the smallest defect type in this dataset
- 54 instances in test split, but many are very small (~3–5px even at 640px)
- Recall=0.426 means 57% of actual mouse bites are missed
- Mitigation: larger image size (e.g., 1280px) or multi-scale inference would help, but not feasible on this CPU

`missing_pad` has lower recall (0.545) despite good precision (0.839):
- Missing pad means absence of material rather than presence, which is harder to detect geometrically
- Model tends to be conservative (high precision, missing some instances)

---

## Reproducibility

To reproduce Experiment A from scratch:

```bash
# 1. Prepare dataset (always same split, seed=42)
C:\Program Files\Python313\python.exe scripts/prepare_dataset.py

# 2. Train
C:\Program Files\Python313\python.exe scripts/train.py --config configs/training_exp_a.yaml

# 3. Evaluate on test split (once only)
C:\Program Files\Python313\python.exe scripts/evaluate.py \
    --model artifacts/models/pcb_yolov8n_640/weights/best.pt \
    --name exp_a
```

Environment: Python 3.13.0, PyTorch 2.14.0+cpu, Ultralytics 8.4.160, seed=42
