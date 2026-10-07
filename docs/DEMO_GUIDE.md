# VisionQC — Demo Guide (Up to Phase 4B)

This guide is designed to help you confidently present the VisionQC project to your teacher. It covers the current implementation state up to **Phase 4B** (FastAPI MLflow Inference Service).

---

## 1. Demo Objective

**What this demo proves:** 
This demo proves that we have successfully built an end-to-end MLOps pipeline for PCB defect detection. It demonstrates that we can take raw data, track experiments and models using MLflow, select a "champion" model (YOLOv8s), and serve that model dynamically through a FastAPI REST API for real-time inference.

*(Note: Do not claim Docker or Airflow are implemented yet, as those are future phases.)*

---

## 2. Demo Preparation Checklist

### **Must Prepare (Before the teacher arrives)**
- [ ] **Open VS Code** to the `VisionQC` folder.
- [ ] **Open 2 separate terminal windows** in VS Code (Terminal 1 for MLflow, Terminal 2 for FastAPI).
- [ ] **Ensure Python 3.13 virtual environment is active** in both terminals (or you are using the explicit Python 3.13 path: `& "C:\Program Files\Python313\python.exe"`).
- [ ] **Prepare Test Image:** Open file explorer to `data/prepared/test/images/` and locate `pcb_defect_007.jpg` so it's ready to upload.
- [ ] **Browser Ready:** Open a browser window with two tabs ready (but don't load the URLs yet).

### **Optional Backup**
- [ ] Have a screenshot of the MLflow dashboard saved just in case the UI fails to load.
- [ ] Have the `reports/evaluation_results_exp_a.json` file open in a VS Code tab as backup evidence of model performance.

---

## 3. Exact Demo Sequence

### **Step 1: Introduction & Architecture**
- **What I do:** Open `README.md` and show the "Repository Structure".
- **What I should see:** The project folders.
- **What I say:** "Sir/Ma'am, this is VisionQC. It detects 6 types of PCB defects using a YOLO model. We structured this as an MLOps project, meaning we track our data, model experiments, and serve the best model via an API."
- **Why it matters:** Establishes that this is an engineering project, not just a Jupyter notebook.

### **Step 2: MLflow Dashboard**
- **What I do:** In Terminal 1, run `& "C:\Program Files\Python313\python.exe" -m mlflow ui --backend-store-uri sqlite:///mlflow.db`. Open `http://localhost:5000` in the browser.
- **What I should see:** The MLflow UI showing experiments and the Model Registry.
- **What I say:** "We use MLflow to track all our training runs. As you can see here, we trained a baseline model and a Candidate (YOLOv8s). Because YOLOv8s achieved a higher mAP50 of 0.81, we registered it here in the Model Registry and tagged it as the 'champion'."
- **Why it matters:** Proves you implemented experiment tracking and model versioning.

### **Step 3: FastAPI Startup**
- **What I do:** In Terminal 2, run `& "C:\Program Files\Python313\python.exe" -m uvicorn src.visionqc.api.main:app --reload`.
- **What I should see:** Terminal logs showing Uvicorn starting and the model loading from MLflow.
- **What I say:** "Now I'll start our inference API. When it boots up, it automatically connects to MLflow, asks for the 'champion' model, and loads it into memory. We don't hardcode the model path."
- **Why it matters:** Demonstrates dynamic model loading (a key MLOps concept).

### **Step 4: Swagger UI & Health Check**
- **What I do:** Open `http://localhost:8000/docs` in the browser. Click `GET /health`, click "Try it out", then "Execute".
- **What I should see:** A 200 OK response with `{"status": "ok", "model_version": 2}`.
- **What I say:** "This is the Swagger UI. Our health endpoint confirms the API is live and correctly loaded Version 2, which is our champion."
- **Why it matters:** Shows API readiness and Swagger documentation.

### **Step 5: Live Prediction**
- **What I do:** Click `POST /predict`. Click "Try it out". Upload `data/prepared/test/images/pcb_defect_007.jpg`. Click "Execute".
- **What I should see:** A 200 OK response with a JSON payload of predictions (bounding boxes, classes).
- **What I say:** "Here I'm sending a real PCB image to the API. The API processes the image, runs the YOLOv8s model inference, and returns the defect predictions in JSON format."
- **Why it matters:** Proves the end-to-end pipeline actually works.

---

## 4. Recommended Demo Order

1. **Briefly introduce the problem** (PCB defect detection).
2. **Show project architecture** (VS Code file tree).
3. **Show model evaluation results** (Briefly mention mAP50 improvement from Phase 4A).
4. **Show MLflow** (Experiments and Model Registry Champion).
5. **Start FastAPI** (Show terminal logs loading the model).
6. **Open Swagger** (API documentation).
7. **Perform a real prediction request** (`/predict` endpoint).
8. **Show prediction response** (JSON output).
9. **Summarize** (How the components connect seamlessly).

---

## 5. MLflow Demo Details

- **Start Command:** `& "C:\Program Files\Python313\python.exe" -m mlflow ui --backend-store-uri sqlite:///mlflow.db`
- **Port:** `http://localhost:5000`
- **What to show:** Click on "Models" at the top to show the `VisionQC-Detector` and point out the `champion` alias on Version 2.
- **What is MLflow doing?** "MLflow acts as our central source of truth for experiments. It logs our training parameters (like epochs, image size) and metrics (mAP, precision). More importantly, it acts as a Model Registry."
- **Why use MLflow?** "Instead of manually copying `.pt` files around and losing track of which model is in production, MLflow lets us programmatically tag the best model as 'champion' so our API can fetch it automatically."

---

## 6. FastAPI Demo Details

- **Start Command:** `& "C:\Program Files\Python313\python.exe" -m uvicorn src.visionqc.api.main:app --reload`
- **Port:** `http://localhost:8000`
- **How model is loaded:** Explain that in `src/visionqc/api/main.py`, the `lifespan` function queries MLflow for `models:/VisionQC-Detector@champion`, downloads the weights dynamically, and loads them into memory exactly once on startup.
- **Processing:** The API receives a file upload, converts it to an RGB Image, passes it to the Ultralytics YOLO model, and parses the raw output into a clean JSON response.

---

## 7. Swagger Demo Details

- **URL:** `http://localhost:8000/docs`
- **What is Swagger?** "Swagger is an interactive documentation page automatically generated by FastAPI. It allows developers to test the API endpoints directly from the browser without needing to write custom code or use tools like Postman."
- **Execution:** Expand the `/predict` endpoint, click "Try it out", use the file picker to choose an image from `data/prepared/test/images/`, and click "Execute". Explain that the JSON response contains the coordinates (`box`) and the defect type (`class`).

---

## 8. End-to-End Architecture Explanation

"Conceptually, our pipeline works like this:
First, our **training pipeline** takes the YOLO dataset and trains a model, logging all metrics and the final weights to **MLflow**. We evaluate the model and tag the best one as 'champion' in the registry.
Second, our **inference pipeline** starts with **FastAPI**. On startup, FastAPI asks MLflow for the champion weights and loads them into memory. When a client sends an image via the `/predict` endpoint, FastAPI feeds it to the model, gets the bounding boxes, and returns them as a JSON response."

---

## 9. Files I Should Show

| File | Why I should show it |
| ---- | -------------------- |
| `src/visionqc/api/main.py` | Shows the FastAPI code, specifically the `lifespan` function fetching the MLflow champion. |
| `configs/training_exp_a.yaml` | Shows how we configure our YOLO training runs cleanly. |
| `README.md` | Proves we have excellent project documentation and an organized MLOps status tracker. |
| `mlflow.db` (in the file tree) | Shows that we are using a local SQLite backend for MLflow tracking. |

---

## 10. What NOT to Show

- ❌ **Do NOT run a training script live.** It will take too long and freeze your CPU.
- ❌ **Do NOT show the raw COCO JSON file.** It's huge and will lag VS Code.
- ❌ **Do NOT talk about Docker or Airflow.** If asked, state "That is scheduled for Phase 5, the current phase finalized the API and Model Registry."

---

## 11. Demo Backup Plan

- **If MLflow UI doesn't start:** Open `reports/evaluation_results_exp_a.json` to prove you evaluated the model.
- **If FastAPI fails to start (Port in use):** Run it on a different port: `& "C:\Program Files\Python313\python.exe" -m uvicorn src.visionqc.api.main:app --port 8080`.
- **If prediction fails during live demo:** Show the `tests/test_api.py` file and run `& "C:\Program Files\Python313\python.exe" -m pytest tests/` to prove it works in automated testing.
- **If internet is down:** Everything is 100% local (SQLite, local models). You do not need internet for this demo.

---

## 12. Exact Commands Cheat Sheet

```bash
# 1. Start MLflow (Terminal 1)
& "C:\Program Files\Python313\python.exe" -m mlflow ui --backend-store-uri sqlite:///mlflow.db

# 2. Start FastAPI (Terminal 2)
& "C:\Program Files\Python313\python.exe" -m uvicorn src.visionqc.api.main:app --reload

# 3. Test API via automated tests (Optional / Backup)
& "C:\Program Files\Python313\python.exe" -m pytest tests/ -v
```

---

## 13. 5-Minute Demo Version

1. Show `src/visionqc/api/main.py` (Point out MLflow model fetching).
2. Start FastAPI.
3. Open Swagger (`localhost:8000/docs`).
4. Upload `pcb_defect_007.jpg` to `/predict`.
5. Show the JSON response.
6. **Say:** "We trained YOLOv8s, tracked it in MLflow, and served it via FastAPI. The API dynamically loads the champion model on startup."

---

## 14. 10–15 Minute Demo Version

Follow the **Exact Demo Sequence (Section 3)** in full. Take the time to show the MLflow UI in the browser, explain the difference between Phase 2 baseline and Phase 4A candidate, and walk through the Swagger UI slowly.

---

## 15. Viva Questions (Short & Memorizable)

- **What is YOLO?** You Only Look Once. It's a real-time object detection model.
- **Why YOLO?** It is fast and accurate, perfect for manufacturing lines where speed matters.
- **What is an epoch?** One complete pass of the training dataset through the algorithm.
- **What is mAP50?** Mean Average Precision at an Intersection over Union (IoU) threshold of 0.5. It measures how accurate our bounding boxes are.
- **Why did you choose YOLOv8s?** It gave us a mAP50 of 0.81, which was significantly better than the baseline YOLOv8n, while still having acceptable CPU latency (346ms).
- **What does MLflow do?** It tracks our training metrics and manages our model versions in a central registry.
- **Why use FastAPI?** It's a modern, fast web framework for Python that auto-generates Swagger documentation.
- **What is an endpoint?** A specific URL where an API can be accessed (like `/predict`).
- **How is this different from simply training a YOLO model?** Training a model is just data science. MLOps (this project) handles the engineering: tracking versions, dynamically loading weights, and serving it reliably via a REST API.

---

## 16. 30-Second Project Explanation

"VisionQC is an MLOps pipeline for detecting PCB manufacturing defects. We processed an annotated dataset, trained YOLO object detection models, and tracked our experiments using MLflow. We then selected the best model, tagged it as our champion in the MLflow registry, and built a FastAPI service that dynamically loads this champion model to perform real-time inference on uploaded images."

---

## 17. 1-Minute Technical Explanation

"Our architecture separates training and inference. For training, we convert COCO annotations to YOLO format, train the model, and log metrics to a local SQLite MLflow backend. For inference, we built a FastAPI application. On startup, the API queries MLflow for the `VisionQC-Detector@champion` model and loads the weights into memory using Ultralytics. When a client hits the `/predict` endpoint, the image is passed to the YOLO model, and we return the parsed JSON bounding boxes. This ensures our production API always uses the best tracked model without hardcoding file paths."

---

## 18. Important Numbers/Details to Memorize

- **Dataset:** 230 images (161 train, 34 val, 35 test)
- **Classes:** 6 types of defects (e.g., missing_pad, mouse_bite, short, etc.)
- **Model:** YOLOv8s (Version 2 is Champion)
- **Image Size:** 640px
- **Best mAP50:** 0.8137
- **CPU Latency:** ~346 ms per image
- **API Port:** 8000
- **MLflow Port:** 5000
