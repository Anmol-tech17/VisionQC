# VisionQC — Dataset Research & Selection

This document records the dataset-selection process for the VisionQC project. It details the datasets researched, compared, and the rationale behind the final selection used for the champion model.

---

## 1. Datasets Researched & Compared

During the initial phases, four primary datasets were considered:

### A. PCB-Defect: An Annotated Dataset for Surface Defect (Rashid 2025)
* **Source:** Real-world FR4 chemical etching defects (Data in Brief, 2025).
* **Size:** 230 high-resolution images.
* **Classes:** 6 classes (`missing_pad`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`).
* **Format:** COCO JSON format.
* **Advantages:** Contains genuine, real-world manufacturing defects with high-resolution imagery.
* **Disadvantages:** Very small dataset size (230 images) which poses a risk of overfitting.

### B. Mixed PCB Defect Dataset (Mendeley Data)
* **Source:** Mendeley Data (DOI: 10.17632/fj4krvmrr5).
* **Size:** 1,386 synthetic images.
* **Classes:** 6 classes (`missing holes`, `mouse bites`, `open circuits`, `shorts`, `spurs`, `spurious copper`).
* **Format:** YOLO format, pre-resized to 640x640.
* **Advantages:** Large dataset size, pre-formatted for YOLO, CC BY 4.0 license. Adds significant domain diversity.
* **Disadvantages:** Synthetic defects generated artificially (derived from PKU-Market-PCB). Intentional augmentations could introduce synthetic biases. `missing holes` (drilled via) is physically distinct from `missing_pad` (copper landing).

### C. PKU-Market-PCB / HRIPCB
* **Source:** Peking University Open Lab / Kaggle.
* **Size:** 1,386 synthetic bare PCB images.
* **Classes:** 6 classes.
* **Disadvantages:** Confirmed to be the original source of the Mendeley dataset. Discarded in favor of the pre-formatted Mendeley dataset.

### D. DeepPCB / SolDef_AI
* **Source:** GitHub / Kaggle.
* **Disadvantages:** Custom annotation formats requiring heavy conversion (DeepPCB) or out-of-scope defects like SMT soldering components (SolDef_AI). Rejected early in the process.

---

## 2. Dataset Evaluation & Experiments

We performed a deep comparative analysis between the **Rashid 2025** dataset and the **Mendeley** dataset.

* **Criteria Used:** Real-world authenticity, class compatibility, leakage risk, and scientific value for model generalization.
* **Leakage Assessment:** We verified that Rashid 2025 (real-world) and Mendeley (synthetic) are independent datasets. Merging them posed a LOW leakage risk, provided the test set remained strictly real-world data.

### Experimental Integration (Phase 3)
In Phase 3, we experimentally created a **V2 Dataset** by safely merging the Mendeley dataset into the Rashid training pipeline. This resulted in an expanded training set of 1,902 images. We trained Experiment B (YOLOv8n) on this merged dataset to evaluate if synthetic augmentations improved robustness.

### Impact of Dataset Choice
While the merged V2 dataset provided more training volume, the synthetic nature of the Mendeley dataset introduced domain gap challenges when evaluated strictly on real-world test images. The pure Rashid dataset, despite being small, provided the most accurate representation of the target real-world distribution.

---

## 3. Final Dataset Decision

**Final Selected Dataset: PCB-Defect (Rashid 2025)**

For the final Phase 4A champion model (YOLOv8s), we reverted to and finalized the **Rashid 2025 dataset (230 real-world images)** as the sole dataset for training and evaluation.

**Reasoning:** The Rashid dataset consists entirely of authentic, real-world PCB manufacturing defects, ensuring the model learns actual physical defect characteristics rather than synthetic Photoshop artifacts introduced by the Mendeley dataset. This maximizes the model's reliability for real-world inference.
