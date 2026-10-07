# VisionQC — Data Card

## 1. Final Selected Dataset (Used for Champion Model)

| Item | Value |
|---|---|
| **Dataset name** | PCB-Defect: An Annotated Dataset for Surface Defect (Rashid 2025) |
| **Source** | Real-world FR4 chemical etching defects (Data in Brief, 2025) |
| **Purpose** | Final training, validation, and testing for the Phase 4A Champion model |
| **Domain** | Bare PCB manufacturing surface defects |
| **Image type** | High-resolution photographs (JPEG), varying aspect ratios |
| **Total images** | 230 |
| **Total annotations**| 1,704 |
| **Number of classes**| 6 |
| **Annotation format**| Original: COCO JSON. Converted: YOLO TXT |

### Defect Classes (Final Dataset)
| Model ID | Class Name | Instances |
|---|---|---|
| 0 | missing_pad | 276 |
| 1 | mouse_bite | 356 |
| 2 | open_circuit | 276 |
| 3 | short | 254 |
| 4 | spur | 296 |
| 5 | spurious_copper | 246 |

### Train / Validation / Test Split (Seed=42)
| Split | Images | Annotations | Ratio |
|---|---|---|---|
| Train | 161 | 1,194 | 70% |
| Validation | 34 | 243 | 15% |
| Test | 35 | 267 | 15% |

*No-leakage verified: Splits are completely distinct at the image level.*

### Preprocessing & Characteristics
* **Bounding Box Analysis:** 96.3% of bounding boxes are smaller than 5% of image dimensions. This small-object dominance required training at a larger resolution (640px).
* **Preprocessing:** COCO to YOLO coordinate conversion, class ID remapping.
* **Augmentation:** Standard Ultralytics YOLO augmentations (hsv_s, hsv_v, flipLR), but `mosaic` was disabled to preserve the spatial context of tiny defects.
* **Strengths:** High authenticity (real-world defects).
* **Weaknesses:** Very small size (230 images) limits broad generalization.

---

## 2. Researched / Experimental Dataset (Used in Phase 3 Only)

| Item | Value |
|---|---|
| **Dataset name** | Mixed PCB Defect Dataset |
| **Source** | Mendeley Data (DOI: 10.17632/fj4krvmrr5) |
| **Purpose** | Experimental training volume expansion (Phase 3, Experiment B) |
| **Domain** | Synthetic bare PCB surface defects (derived from PKU-Market-PCB) |
| **Image type** | Synthetic images with intentional augmentations, pre-resized to 640x640 |
| **Total images** | 1,386 |
| **Number of classes**| 6 (`missing holes`, `mouse bites`, `open circuits`, `shorts`, `spurs`, `spurious copper`) |
| **Annotation format**| YOLO format |
| **Licensing** | CC BY 4.0 |

### Preprocessing & Characteristics
* **Integration:** Safely merged into the training pipeline in Phase 3 to form a "V2 Dataset" (1,902 total training images). Class `missing holes` was mapped distinctively from `missing_pad` to avoid semantic collision.
* **Strengths:** Large volume of data, pre-formatted for YOLO, adds domain diversity.
* **Weaknesses:** Defects are synthetic and artificially generated, which introduces bias and doesn't perfectly mirror real-world chemical etching anomalies.

---

## 3. Dataset Comparison & Final Decision

| Criterion | PCB-Defect (Rashid 2025) | Mixed PCB Defect (Mendeley) |
|---|---|---|
| **Authenticity** | Real-world | Synthetic |
| **Volume** | Low (230 images) | High (1,386 images) |
| **Defect Types** | Genuine physical anomalies | Photoshop-generated artifacts |
| **Selection Status**| **FINAL SELECTED DATASET** | **REJECTED FOR FINAL MODEL** |

**Final Decision:** The **PCB-Defect (Rashid 2025)** dataset was ultimately selected as the exclusive dataset for training the final Phase 4A champion model. 
**Reason:** Despite its small size, its real-world authenticity provided significantly higher scientific and practical value for detecting genuine manufacturing defects compared to the synthetic artifacts present in the Mendeley dataset.
