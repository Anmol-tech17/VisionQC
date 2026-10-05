# VisionQC — Phase 2 Report

*Author: MLOps Course Project*  
*Phase 2: Real ML Inference Layer*

---

## 1. Executive Summary

Phase 2 replaces the Phase 1 COCO annotation lookup with a **real trained YOLOv8n object detector** fine-tuned on 230 annotated PCB defect images. The pipeline now performs genuine ML inference, returning actual model confidence scores and predicted bounding boxes rather than stored ground-truth annotations.

Two training experiments were conducted: a baseline (320px, 20 epochs) and Experiment A (640px, 40 epochs with PCB-specific augmentation). The final model is selected objectively based on validation metrics, then evaluated once on the held-out test split.

---

## 2. Before Phase 2

Phase 1 used `COCOAnnotationPredictor`:

```python
# Phase 1 - NOT ML inference
detections = annotation_index[uploaded_filename]   # dictionary lookup
```

This looked up ground-truth bounding boxes from the COCO JSON file using the uploaded filename as a key. It was correct only for images already in the dataset, returned pre-labelled ground truth instead of predictions, and confidence was always 1.0 (not a model score).

---

## 3. What Changed

| Component | Phase 1 | Phase 2 |
|---|---|---|
| Inference | COCO annotation lookup | Trained YOLOv8n model |
| Confidence | Always 1.0 (ground truth) | Real model probability [0,1] |
| Bounding boxes | Pre-stored annotations | Model-predicted coordinates |
| Works on unseen images | No | Yes |
| Prediction label | "ground_truth" | "yolo_model" |
| Model path | None | `artifacts/models/pcb_yolov8n_640/weights/best.pt` |
| Training pipeline | None | `scripts/train.py` |
| Evaluation pipeline | None | `scripts/evaluate.py` |

---

## 4. Dataset

| Item | Value |
|---|---|
| Total images | 230 |
| Total annotations | 1,704 |
| Defect classes | 6 |
| Annotation format | COCO JSON → YOLO TXT |
| Random seed | 42 |

**Splits (image-level, no leakage):**

| Split | Images | Annotations | Ratio |
|---|---|---|---|
| Train | 161 | 1,194 | 70% |
| Val | 34 | 243 | 15% |
| Test | 35 | 267 | 15% |

---

## 5. Dataset Preparation

The COCO annotation file was converted to YOLO format:

1. Exclude category 0 (`detecting-pcb-defects` — root label, not a defect)
2. Map COCO category IDs to contiguous 0-based model IDs
3. Convert bounding boxes: `[x_min, y_min, w, h]` → `[cx, cy, w, h]` (normalised 0–1)
4. Perform image-level random split (seed=42)
5. Write `.txt` label files to `data/prepared/{split}/labels/`
6. Write `data/prepared/dataset.yaml`

---

## 6. Class Mapping

| COCO Category ID | Class Name | Model ID |
|---|---|---|
| 0 | detecting-pcb-defects (root) | **excluded** |
| 1 | missing_pad | 0 |
| 2 | mouse_bite | 1 |
| 3 | open_circuit | 2 |
| 4 | short | 3 |
| 5 | spur | 4 |
| 6 | spurious_copper | 5 |

---

## 7. Dataset Quality Audit

A dedicated audit (`scripts/audit_dataset.py`) was run before experiments.

**Critical finding:**

| Image size | Median defect size |
|---|---|
| 320 px | **9 × 9 px** |
| 416 px | 12 × 12 px |
| 640 px | **19 × 19 px** |

96.3% of all bounding boxes are smaller than 5% of image width or height. This means at 320px the majority of defects are near or below YOLO's reliable detection threshold for small objects. This directly explains the poor recall for `mouse_bite`, `spur`, and `missing_pad` in the baseline.

**Annotation quality:**
- Zero invalid YOLO coordinates
- Zero empty label files
- All 6 classes present in all splits
- No image leakage between splits

---

## 8. Model Architecture

**YOLOv8n (nano)**

- YOLO = "You Only Look Once" — single-pass object detector
- Predicts bounding boxes and class probabilities in one forward pass
- "nano" = smallest variant: ~3.2M parameters, 8.1 GFLOPs
- Base weights: pretrained on COCO dataset (80 classes) — transfer learning
- Fine-tuned on PCB defect dataset (6 classes)
- Output: bounding boxes as [x1, y1, x2, y2] + class + confidence

**Why YOLOv8n?**
- Fastest inference on CPU (i3-8130U ~128ms/image)
- Proven transfer learning from COCO
- Well-maintained ultralytics library (Python 3.13 compatible)
- Simple Python API for FastAPI integration

---

## 9. Training

### Baseline (training.yaml)

| Parameter | Value |
|---|---|
| Image size | 320 px |
| Epochs | 20 / 20 |
| Batch | 4 |
| Patience | 5 |
| Seed | 42 |
| Duration | 16.1 minutes |
| Augmentation | Ultralytics defaults |

Val metrics (best epoch): Precision=0.182, Recall=0.181, mAP@0.5=0.160

### Experiment A (training_exp_a.yaml)

| Parameter | Value | Rationale |
|---|---|---|
| Image size | 640 px | Median defect grows from 9px to 19px |
| Epochs | 40 | More learning cycles |
| Patience | 10 | Allow longer plateaus |
| Mosaic | OFF | Crops destroy tiny defect context |
| Rotation/shear | OFF | PCB orientation is meaningful |
| flipLR | 0.5 | Harmless symmetry |
| Colour aug | Mild (s=0.3, v=0.3) | Realistic lighting variation |

---

## 10. Evaluation Methodology

1. **Train** on train split
2. **Monitor** validation metrics during training (early stopping)
3. **Select best model** based on validation mAP@0.5
4. **Evaluate once** on held-out test split — test set never used during training
5. **Report** only test-split metrics as final metrics

---

## 11. Baseline Test Results

| Metric | Value |
|---|---|
| Precision | 0.2470 |
| Recall | 0.1668 |
| mAP@0.5 | 0.1650 |
| mAP@0.5:0.95 | 0.0670 |

| Class | Precision | Recall | AP@0.5 |
|---|---|---|---|
| missing_pad | 0.1123 | 0.0227 | 0.0510 |
| mouse_bite | 0.0000 | 0.0000 | 0.0141 |
| open_circuit | 0.4386 | 0.2553 | 0.2422 |
| short | 0.5243 | 0.1951 | 0.1987 |
| spur | 0.0000 | 0.0000 | 0.0221 |
| spurious_copper | 0.4066 | 0.5278 | 0.4621 |

Inference latency: mean=128ms, P95=267ms (CPU)

---

## 12. Experiment A Results (640px, PCB-conservative augmentation)

**Val metrics (best epoch, val split):**

| Metric | Baseline | Exp A | Delta |
|---|---|---|---|
| Precision | 0.1820 | **0.7456** | +0.564 |
| Recall | 0.1806 | **0.7004** | +0.520 |
| mAP@0.5 | 0.1604 | **0.7400** | +0.580 |
| mAP@0.5:0.95 | 0.0628 | **0.3698** | +0.307 |

---

## 13. Final Model Selection

**Winner: Experiment A (640px)**  
**Model path:** `artifacts/models/pcb_yolov8n_640/weights/best.pt`

Selected by validation mAP@0.5: Exp A = 0.7400 vs Baseline = 0.1604

**Final test metrics (held-out test split, evaluated once):**

| Metric | Baseline | **Final (Exp A)** | Delta |
|---|---|---|---|
| Precision | 0.2470 | **0.8192** | +0.5722 |
| Recall | 0.1668 | **0.6954** | +0.5286 |
| mAP@0.5 | 0.1650 | **0.7527** | +0.5877 |
| mAP@0.5:0.95 | 0.0670 | **0.3610** | +0.2940 |

Inference latency (CPU): mean=185ms, P95=299ms

---

## 14. Confidence Scores

**What confidence means:**

When the model predicts a defect, it outputs a probability score between 0 and 1:
- `0.91` means the model is 91% confident this region contains a defect of that class
- Scores below the threshold (default 0.25) are filtered out
- The score comes from the neural network's sigmoid/softmax output — it is NOT a ground-truth label

**Baseline observation:** The model tends to output lower confidences on this small dataset. Some correctly detected defects appear at 0.3–0.5 confidence. This is expected for a model trained on only 161 images.

---

## 15. Inference Pipeline

```
User uploads PCB image (JPEG/PNG)
        ↓
FastAPI /predict endpoint
        ↓
Image validation (PIL verify)
        ↓
YOLOPredictor.predict(image_bytes)
        ↓
PIL image → ultralytics YOLO model
        ↓
Model forward pass (YOLOv8n)
        ↓
NMS (IoU threshold = 0.45)
        ↓
Confidence filtering (threshold = 0.25)
        ↓
Convert xyxy → xywh bbox format
        ↓
[{class_name, class_id, confidence, bbox, source:"yolo_model"}]
        ↓
draw_detections() → annotated JPEG bytes
        ↓
base64 encode
        ↓
JSON response to browser
        ↓
UI renders annotated image + detection table
```

---

## 16. FastAPI Endpoints

### GET /health

Returns:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "pcb_yolov8n_640",
  "confidence_threshold": 0.25,
  "dataset_valid": true,
  "uptime_seconds": 42.1,
  "version": "0.2.0"
}
```

### POST /predict

Accepts: JPEG or PNG image upload  
Returns:
```json
{
  "detections": [
    {"class_name": "short", "class_id": 3, "confidence": 0.87, "bbox": [x,y,w,h], "source": "yolo_model"}
  ],
  "count": 1,
  "annotated_image": "<base64 JPEG>",
  "prediction_type": "yolo_model",
  "confidence_threshold": 0.25,
  "model_name": "pcb_yolov8n_640"
}
```

---

## 17. Testing

The test suite (`tests/test_api.py`) contains 20 tests:
- Health endpoint structure and model_name field
- Model-dependent tests auto-skip if best.pt doesn't exist
- Confidence range validation (0.0–1.0)
- Class name validation (must be one of 6 defect classes)
- prediction_type == "yolo_model" assertion
- Rejection of non-image uploads
- Dataset validator failure on wrong path

---

## 18. Reproducibility

| Item | Value |
|---|---|
| Random seed | 42 |
| Python | 3.13.0 |
| PyTorch | 2.14.0+cpu |
| Ultralytics | 8.4.160 |
| Dataset split seed | 42 |
| Split method | Image-level random shuffle |

To reproduce:
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare dataset (same split every time, seed=42)
python scripts/prepare_dataset.py

# 3. Train baseline
python scripts/train.py

# 4. Train Experiment A
python scripts/train.py --config configs/training_exp_a.yaml

# 5. Evaluate
python scripts/evaluate.py --model artifacts/models/pcb_yolov8n_640/weights/best.pt --name exp_a
```

---

## 19. GitHub Readiness

Items committed: source code, configs, tests, docs, `requirements.txt`, `README.md`  
Items NOT committed (in `.gitignore`): raw dataset, `data/prepared/`, `artifacts/` (model weights), `.venv/`, Python caches, `runs/`, `*.pt`

---

## 20. Limitations

1. **Small dataset (230 images):** Generalisation to unseen PCBs is uncertain
2. **96% small objects:** Even at 640px, minimum defects are 3px — difficult to detect reliably
3. **CPU-only training:** Limited number of experiments feasible
4. **Single manufacturing environment:** Domain shift risk on different boards
5. **No negative examples:** Model hasn't seen defect-free PCBs
6. **Baseline-only for course:** Two additional candidate models planned for MLflow phase

---

## 21. Course Requirement Mapping

| Requirement | Status | Evidence |
|---|---|---|
| Git/GitHub | Prepared | Repository structure + .gitignore |
| Fixed seed | Implemented | seed=42 in all configs |
| Baseline model | Phase 2 | YOLOv8n trained, evaluated |
| Candidate models | Partial | Exp A trained; more planned in MLflow phase |
| DVC | Later | Not implemented |
| MLflow | Later | Not implemented |
| Airflow | Later | Not implemented |
| FastAPI | Implemented | /health + /predict working |
| Docker | Later | Not implemented |
| Tests | Implemented | 20 tests in test_api.py |
| Monitoring | Later | Not implemented |
| Explainability | Later | Not implemented |
| Model card | Implemented | docs/MODEL_CARD.md |
| Data card | Implemented | docs/DATA_CARD.md |
| AWS | Later | Not implemented |

---

## 22. Next Phase Plan

The Phase 3 MLOps infrastructure will wrap around this Phase 2 ML layer:

```
DVC
  → version data/prepared/ and artifacts/
MLflow
  → track experiments (baseline, exp_a, exp_b, ...)
  → model registry + promotion
Airflow
  → orchestrate: prepare → train → evaluate → gate → deploy
Docker
  → containerise the FastAPI app
GitHub Actions
  → CI/CD: lint → test → build → push image
Prometheus/Grafana
  → runtime monitoring (latency, prediction distribution)
AWS/SageMaker
  → cloud training + deployment
```

---

## 23. 2-Minute Viva Explanation

"VisionQC is an automated PCB defect detection system. We have a dataset of 230 PCB images annotated with 1,704 bounding boxes across 6 defect types. In Phase 2, we trained a YOLOv8n model using transfer learning from COCO pretrained weights. YOLOv8n is a real-time object detector that predicts bounding boxes and class labels in a single forward pass. We found that 96% of defects are very small — at 320 pixels, the median defect is only 9×9 pixels, which is too small for reliable detection. So we increased the image size to 640px for Experiment A, which makes the same defect appear as 19×19 pixels. The trained model is served through a FastAPI REST API and a browser UI. We're currently at Phase 2 — the ML layer. In later phases we'll add DVC for dataset versioning, MLflow for experiment tracking, Docker for containerisation, and Airflow for pipeline orchestration."

---

## 24. Likely Viva Questions

**Q: Why object detection rather than classification?**  
A: Classification would only tell us "defective or not." Object detection gives us the location and type of each individual defect, which is essential for automated repair guidance.

**Q: What is a bounding box?**  
A: A rectangle that tightly encloses a detected object, defined by [x, y, width, height]. In YOLO format, coordinates are normalised to [0,1] relative to image size.

**Q: What is confidence?**  
A: The model's estimated probability that a detected region contains an object of the predicted class. We filter out detections below 0.25 to reduce false positives.

**Q: What is mAP?**  
A: Mean Average Precision. For each class, we compute the area under the precision-recall curve (AP). mAP averages this across all classes. mAP@0.5 uses IoU threshold=0.5 to decide if a detection matches ground truth.

**Q: What is IoU?**  
A: Intersection over Union — the area of overlap between predicted and ground-truth bounding boxes, divided by their union. IoU≥0.5 means the prediction is "good enough" to count as a true positive.

**Q: What is precision?**  
A: Of all detections the model made, what fraction were correct? `TP / (TP + FP)`

**Q: What is recall?**  
A: Of all real defects in the image, what fraction did the model find? `TP / (TP + FN)`. For safety-critical manufacturing, high recall is especially important.

**Q: Why is recall low for mouse_bite and spur?**  
A: These are among the smallest defect types. At 320px the model can barely resolve them. Experiment A (640px) is expected to improve recall for these classes.

**Q: Why 230 images?**  
A: This is the available annotated dataset. 230 is small for object detection — production systems typically need thousands of images per class. This limits generalisation.

**Q: What is overfitting?**  
A: When a model learns the training data too specifically and performs poorly on new data. With 161 training images, overfitting is a real risk. We use early stopping and pretrained weights to mitigate this.

**Q: Why train/val/test split?**  
A: Train: for model fitting. Val: for training decisions (hyperparameters, early stopping) without contaminating the final measurement. Test: used exactly once for the unbiased final metric.

**Q: Why fixed seed?**  
A: Reproducibility — the same seed produces the same dataset split and the same training dynamics, allowing others to verify results.

**Q: Why YOLOv8 not Faster-RCNN?**  
A: YOLOv8 is much faster for inference (single-pass), better suited for real-time API use, and the ultralytics library makes training on custom datasets much simpler.

**Q: Why will we use DVC?**  
A: DVC versions large data files (dataset, model weights) that are too big for Git. It lets us track what dataset version was used for each experiment.

**Q: Why will we use MLflow?**  
A: MLflow logs every experiment (hyperparameters, metrics, artifacts) in a queryable database. It provides model registry with versioning and promotion workflows.

**Q: Why Docker?**  
A: Docker packages the FastAPI app and all dependencies into a portable container, ensuring the same environment runs on any machine or cloud server.

**Q: What happens if the image is from a different PCB?**  
A: Domain shift — the model may perform poorly on PCBs with different colours, layouts, or defect appearances than the training data. This is a known limitation.

**Q: What are false positives and false negatives in this context?**  
A: False positive: model reports a defect where there is none (unnecessary rework, cost). False negative: model misses a real defect (defective product shipped to customer, more serious).

**Q: Is this model production-ready?**  
A: No. 230 training images is insufficient for reliable production use. It needs more data, GPU training, validation on real production boards, and human oversight.
