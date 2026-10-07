# VisionQC — Model Card

This Model Card details all models trained and evaluated during the VisionQC project.

---

## 1. Model Comparison Table

| Feature | Baseline Model | Experiment A | Experiment B | Candidate / Champion Model |
|---|---|---|---|---|
| **Model Name** | YOLOv8n_Baseline | YOLOv8n_ExpA | YOLOv8n_ExpB | **VisionQC-Detector (Champion)** |
| **Architecture** | YOLOv8n (nano) | YOLOv8n (nano) | YOLOv8n (nano) | **YOLOv8s (small)** |
| **Framework** | Ultralytics 8.4.160 | Ultralytics 8.4.160 | Ultralytics 8.4.160 | **Ultralytics 8.4.160** |
| **Dataset Used** | Rashid (230 imgs) | Rashid (230 imgs) | Rashid + Mendeley V2 (1,902 imgs) | **Rashid (230 imgs)** |
| **Input Size** | 320px | 640px | 640px | **640px** |
| **Epochs** | 20 | 40 | 40 | **40** |
| **Batch Size** | 4 | 4 | 4 | **4** |
| **Environment** | CPU | CPU | CPU | **CPU** |
| **Purpose** | Establish baseline | Improve small defect recall | Test synthetic data volume | **Final Performance Upgrade** |
| **Selection Status**| Superseded | Superseded | Rejected | **FINAL SELECTED MODEL** |

---

## 2. Detailed Model Profiles

### A. Baseline Model (YOLOv8n_Baseline)
* **Dataset:** Rashid (Real-world only)
* **Training Setup:** 320px, 20 epochs, mosaic enabled (default).
* **Test mAP50:** 0.1650
* **Test mAP50-95:** 0.0670
* **Precision / Recall:** 0.2470 / 0.1668
* **Inference Latency:** 128 ms (CPU)
* **Strengths:** Fast to train and run.
* **Weaknesses:** Failed to detect small defects due to 320px resolution shrinking defects below detectable thresholds.

### B. Experiment A (YOLOv8n_ExpA)
* **Dataset:** Rashid (Real-world only)
* **Training Setup:** 640px, 40 epochs, mosaic OFF.
* **Test mAP50:** 0.7400 (val)
* **Test mAP50-95:** 0.3698 (val)
* **Precision / Recall:** 0.7456 / 0.7004 (val)
* **Inference Latency:** ~185 ms (CPU)
* **Strengths:** Massive performance jump simply by increasing resolution to 640px and disabling mosaic augmentation.
* **Weaknesses:** Still a nano architecture, leaving room for representation improvement.

### C. Experiment B (YOLOv8n_ExpB)
* **Dataset:** V2 Dataset (Rashid + Mendeley synthetic data)
* **Training Setup:** 640px, 40 epochs.
* **Evaluation:** Evaluated strictly on the held-out real-world test set.
* **Metrics:** *(Logged in MLflow during Phase 3, explicit metrics omitted here as it was superseded by Phase 4A)*
* **Strengths:** Leveraged a much larger training volume (1,902 images).
* **Weaknesses:** Synthetic data introduced domain gap issues when evaluating on purely real-world test images. Rejected in favor of pure real-world training.

### D. Best/Final Model: VisionQC-Detector (YOLOv8s)
* **Dataset:** Rashid (Real-world only)
* **Training Setup:** 640px, 40 epochs, YOLOv8s pretrained base.
* **Test mAP50:** **0.8137**
* **Test mAP50-95:** **0.4294**
* **Precision / Recall:** **0.8284** / **0.7863**
* **Inference Latency:** 346.0 ms (CPU)
* **Model Size:** YOLOv8 small variant (~11.2M parameters)

---

## 3. Best/Final Model Decision

**Selected Model:** VisionQC-Detector (YOLOv8s)

**Reason for Selection:** The YOLOv8s architecture was selected as the final Champion because it provided a substantial and definitive improvement in test mAP50 (0.8137 vs 0.7527) and overall precision/recall compared to the nano variants, proving highly capable of identifying tiny real-world PCB defects.

**Artifact Location:** The model is tracked in MLflow and registered under the name `VisionQC-Detector` with the `champion` alias resolving to Version 2. It is served dynamically by the FastAPI inference service.
