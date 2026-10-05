# VisionQC — Data Card

## Dataset Overview

| Item | Value |
|---|---|
| Dataset name | PCB-Defect: An Annotated Dataset for Surface Defect |
| Annotation format | COCO JSON (bounding-box object detection) |
| Total images | 230 |
| Total annotations | 1,704 |
| Number of defect classes | 6 |
| Image format | JPEG |

## Defect Classes

| COCO Category ID | Class Name | Model ID (0-based) |
|---|---|---|
| 0 | detecting-pcb-defects (root, **excluded**) | — |
| 1 | missing_pad | 0 |
| 2 | mouse_bite | 1 |
| 3 | open_circuit | 2 |
| 4 | short | 3 |
| 5 | spur | 4 |
| 6 | spurious_copper | 5 |

## Class Distribution (All Images)

| Class | Instances | Train | Val | Test | % of Total |
|---|---|---|---|---|---|
| missing_pad | 276 | 193 | 39 | 44 | 16.2% |
| mouse_bite | 356 | 252 | 50 | 54 | 20.9% |
| open_circuit | 276 | 191 | 38 | 47 | 16.2% |
| short | 254 | 177 | 36 | 41 | 14.9% |
| spur | 296 | 208 | 43 | 45 | 17.4% |
| spurious_copper | 246 | 173 | 37 | 36 | 14.4% |
| **Total** | **1,704** | **1,194** | **243** | **267** | **100%** |

**Note:** Distribution is relatively balanced (14.4%–20.9%). Class imbalance is NOT the primary performance bottleneck.

## Train / Validation / Test Split

| Split | Images | Annotations | Ratio |
|---|---|---|---|
| Train | 161 | 1,194 | 70% |
| Validation | 34 | 243 | 15% |
| Test | 35 | 267 | 15% |
| **Total** | **230** | **1,704** | **100%** |

**Split strategy:** Image-level random split (a source image never appears in two splits).  
**Random seed:** 42 (deterministic/reproducible).  
**No-leakage verified:** train ∩ val = ∅, train ∩ test = ∅, val ∩ test = ∅.

## Dataset Quality Audit Results

A full quality audit was performed (`scripts/audit_dataset.py`) before any training.

| Check | Result |
|---|---|
| Invalid YOLO coordinates | 0 |
| Empty label files | 0 |
| Images missing labels | 0 |
| Orphan label files | 0 |
| Leakage between splits | None (verified) |
| All 6 classes in train | Yes |

## Bounding Box Size Analysis

**This is the most important dataset characteristic for model design.**

| Statistic | Width (normalised) | Height (normalised) |
|---|---|---|
| Minimum | 0.0053 | 0.0052 |
| Mean | 0.0406 | 0.0407 |
| Median | 0.0298 | 0.0303 |
| Maximum | 0.4898 | 0.4036 |
| **Boxes < 5% dim** | **96.3%** | — |

**Effect on model at different image sizes:**

| Image size | Median defect size (px) | Min defect (px) |
|---|---|---|
| 320 × 320 | ~9 × 9 | ~2 × 2 |
| 416 × 416 | ~12 × 12 | ~2 × 2 |
| 640 × 640 | ~19 × 19 | ~3 × 3 |

At 320px, the majority of defects occupy fewer than 100 pixels — near the lower limit of what YOLO can reliably detect. This finding directly motivated Experiment A (640px image size).

## Image Dimensions

Sample of 30 training images:

| Metric | Width | Height |
|---|---|---|
| Minimum | 1,540 px | 1,438 px |
| Maximum | 3,844 px | 3,857 px |
| Mean | ~2,530 px | ~2,825 px |
| Aspect ratio range | 0.66 – 1.55 | mean ≈ 0.95 |

Images are high-resolution close-up photographs of PCB boards. They are resized to the training image size (320 or 640px) by ultralytics during training.

## Preprocessing Applied

1. Exclude root COCO category (ID 0: `detecting-pcb-defects`) — not a defect type
2. Map COCO category IDs (1–6) to contiguous model IDs (0–5)
3. Convert bounding boxes from COCO `[x_min, y_min, w, h]` to YOLO normalised `[cx, cy, w, h]`
4. Clamp coordinates to [0, 1] (all were valid — no clamping needed in practice)
5. Perform image-level random split (seed=42, 70/15/15)
6. Copy images to `data/prepared/{split}/images/`
7. Write YOLO `.txt` label files to `data/prepared/{split}/labels/`
8. Write `data/prepared/dataset.yaml`

## Dataset Preparation Procedure

```bash
# Always use Python 3.13 (ultralytics dependency)
C:\Program Files\Python313\python.exe scripts/prepare_dataset.py
```

The output is deterministic: same seed → same split → same label files.

## Known Limitations

1. **Small dataset (230 images):** Object detection models typically need thousands of images per class. Expect limited generalisation.

2. **Nearly all objects are small:** 96.3% of bounding boxes are smaller than 5% of image dimensions. This is unusual and challenging for standard detectors. Even at 640px, some defects are only 3–4 pixels across.

3. **Single manufacturing environment:** All images appear to come from one type of PCB board and one inspection setup. The model may not generalise to different board designs, lighting conditions, or camera setups (domain shift).

4. **No background/negative images:** The dataset contains only defective PCBs. The model has not been trained on defect-free PCBs, so the false positive rate on normal boards is unknown.

5. **Mild class imbalance:** `mouse_bite` (20.9%) and `spur` (17.4%) are slightly more frequent than `short` (14.9%) and `spurious_copper` (14.4%). Effect on detection quality is minor compared to the small-object problem.

6. **Annotation quality not independently verified:** Bounding box tightness and class label accuracy were not assessed for inter-annotator agreement.

## Potential Bias

- Dataset may not represent the full range of PCB designs manufactured in production
- Lighting, zoom level, and camera angle are not documented
- All images from a single annotator or annotation tool (quality/consistency unknown)

## Intended Use

- Training and evaluating PCB surface defect detection models for the VisionQC project
- Educational demonstration of object detection in manufacturing (MLOps course project)
- NOT intended for deployment in safety-critical manufacturing systems without additional large-scale validation

## Out-of-Scope Use

- Real-time safety-critical inspection without human oversight
- General manufacturing defect detection outside the PCB domain
- Production use without additional validation data representative of actual production

## Dataset Location

```
(local) MLOPS CP/
  PCB-Defect An Annotated Dataset for Surface Defect/
    PCB-Defect An Annotated Dataset for Surface Defect/
      PCB_Defect/PCB_Defect/
        annotation/_annotations.coco.json
        images/
          pcb_defect_001.jpg ... pcb_defect_230.jpg
```

The raw dataset is **not committed to Git**. It will be versioned with DVC in a later phase.

The prepared YOLO format dataset (`data/prepared/`) is also excluded from Git (regenerated by `scripts/prepare_dataset.py`).
