# VisionQC — Practical Demonstration Guide
**Project:** VisionQC: Automated Manufacturing Defect Inspection  
**Target Environment:** Windows 10/11 + WSL2 (`Ubuntu-D`, Docker on D: drive, CPU-only)  
**Authoritative Location:** `docs/DEMO_GUIDE.md`  

---

## 1. Pre-Demo Checklist

Complete these checks 5–10 minutes before the demonstration:

- [ ] **Repository Directory:** Ensure you are in `C:\Users\dell\OneDrive\Desktop\MLOPS CP\VisionQC`.
- [ ] **Python Interpreter:** Confirm Python 3.13 is available (`& "C:\Program Files\Python313\python.exe" --version`).
- [ ] **Git Synchronization:** Confirm working tree is clean and synchronized (`git status`).
- [ ] **Champion Model Weights:** Confirm `artifacts/models/pcb_yolov8n/weights/best.pt` exists (~21.5 MB).
- [ ] **MLflow Database:** Confirm `mlflow.db` exists at project root.
- [ ] **Sample Images Ready:** Locate test images in File Explorer at `data/prepared/test/images/` (specifically `pcb_defect_007.jpg`).
- [ ] **Docker Engine in WSL:** Confirm Ubuntu-D Docker daemon is running:
  ```powershell
  wsl.exe -d Ubuntu-D -u root -e docker ps
  ```
  *(If WSL was sleeping, this command awakens it).*
- [ ] **Browser Prepared:** Open your browser with clean tabs ready for the demonstration URLs.

---

## 2. Exact Startup Commands

VisionQC supports two demonstration modes. **Option A (Python Native)** is fastest for immediate GUI and inference testing. **Option B (Docker Stack)** demonstrates full containerized MLOps architecture.

### Option A: Fast Local Demonstration (Windows PowerShell)

Run these commands in Windows PowerShell:

```powershell
# 1. Open project directory
Set-Location "C:\Users\dell\OneDrive\Desktop\MLOPS CP\VisionQC"

# 2. Launch FastAPI Inference Server & Web GUI
& "C:\Program Files\Python313\python.exe" main.py
```
*The server will boot, automatically load the Champion model from `mlflow.db`, and serve the GUI at `http://localhost:8000`.*

---

### Option B: Full Containerized Stack (Ubuntu-D WSL)

Open a WSL terminal (or run via PowerShell using `wsl.exe`):

```bash
# In Ubuntu-D WSL:
cd "/mnt/c/Users/dell/OneDrive/Desktop/MLOPS CP/VisionQC"

# 1. Ensure shared bridge network exists
docker network create visionqc-net 2>/dev/null || true

# 2. Start Inference API container (Port 8000)
docker compose up -d

# 3. Start Prometheus (Port 9090) and Grafana (Port 3000)
docker compose -f docker-compose.monitoring.yml up -d

# 4. (Optional) Start Airflow pipeline orchestrator (Port 8080)
docker compose -f docker-compose.airflow.yml up -d
```

---

## 3. Verified Demonstration URLs

| Service / Interface | Local URL | Default Credentials | Purpose |
|---|---|---|---|
| **Industrial Web GUI** | `http://localhost:8000` | *None* | Primary user interface for PCB inspection. |
| **FastAPI Swagger Docs** | `http://localhost:8000/docs` | *None* | Interactive OpenAPI explorer and schema. |
| **API Health Status** | `http://localhost:8000/health` | *None* | Verifies model loading & version alias. |
| **Prometheus Metrics** | `http://localhost:8000/metrics` | *None* | Raw Prometheus text-format scrape output. |
| **Prometheus Targets** | `http://localhost:9090/targets` | *None* | Proves API scraper target is `UP`. |
| **Grafana Monitoring** | `http://localhost:3000` | `admin` / `admin` | Live 9-panel telemetry dashboard. |
| **Apache Airflow** | `http://localhost:8080` | `admin` / `admin` | DAG visualization & pipeline run history. |

---

## 4. Recommended Demonstration Sequence (8–10 Minutes)

Follow this structured order for maximum impact:

### Step 1: Problem & System Introduction (1 min)
* **Action:** Open `http://localhost:8000` in the browser.
* **Talking Point:** "VisionQC is an automated manufacturing defect inspection system. In electronics manufacturing, manual optical inspection of PCBs is error-prone. VisionQC uses a YOLOv8 object detection model to detect 6 surface defect types in real time, built within a complete MLOps lifecycle."

### Step 2: Overview Dashboard & System Telemetry (1 min)
* **Action:** Highlight the KPI cards on the **Dashboard** tab.
* **Show:** API status (Online), Model status (Loaded), Active version (`VisionQC-Detector v2`), and 6 Defect Classes.
* **Talking Point:** "Our dashboard pulls live metadata from our FastAPI health service, verifying that the registered Champion model is loaded into memory."

### Step 3: Live PCB Defect Inspection (2.5 mins)
* **Action:** Click the **Inspection** tab.
  1. Drag and drop `data/prepared/test/images/pcb_defect_007.jpg` into the upload zone.
  2. Point out the instant thumbnail preview and file size badge.
  3. Click **Run Inspection**.
* **Show:**
  * Animated loading spinner during CPU inference.
  * Output image with color-coded bounding boxes drawn server-side via OpenCV.
  * Overlay badge showing the defect count (e.g., 6–8 defects found).
  * Latency counter showing server inference time (~250–350 ms).
  * Detection table displaying defect class chips (`mouse_bite`, `spur`, `short`) and confidence progress bars.
* **Talking Point:** "This is actual model inference executing on CPU. The bounding boxes and confidence scores are calculated dynamically by the model, not read from ground truth annotations."

### Step 4: Model Architecture & Experiment Tracking (1.5 mins)
* **Action:** Click the **Model Info** tab in GUI, then briefly show `docs/MODEL_CARD.md`.
* **Talking Point:** "We evaluated multiple architectures using MLflow. Our baseline YOLOv8n @ 320px achieved only 0.165 mAP50 due to tiny defect sizes. By increasing resolution to 640px and adopting YOLOv8s, our champion model achieved **0.8137 mAP50**, registered as `VisionQC-Detector@champion`."

### Step 5: Data Versioning with DVC (1 min)
* **Action:** In VS Code / terminal, show `data/raw/rashid.dvc` and `.dvc/config`.
* **Talking Point:** "Raw datasets are versioned with DVC so that multi-gigabyte image data is never committed to Git, maintaining reproducible lineage with remote storage on D:."

### Step 6: Pipeline Orchestration with Airflow (1 min)
* **Action:** Open `http://localhost:8080` (or show `dags/visionqc_pipeline.py`).
* **Talking Point:** "We orchestrated the training pipeline using Apache Airflow. The DAG has 6 stages: validate inputs, prepare dataset, train candidate, evaluate, compare against baseline, and record results. It runs in safe smoke-test mode by default so full training isn't triggered accidentally."

### Step 7: Continuous Telemetry with Prometheus & Grafana (1.5 mins)
* **Action:**
  1. Open `http://localhost:8000/metrics` to show raw metrics.
  2. Open `http://localhost:3000` → Dashboards → **VisionQC API Monitoring**.
* **Show:** Live panels for Total API Requests, Request Rate, Latency Quantiles (p50/p95), HTTP status codes, and YOLO inference duration.
* **Talking Point:** "Our FastAPI service is instrumented with `prometheus_client`. Prometheus scrapes metrics every 10 seconds, and Grafana visualizes throughput and latency distributions. High-cardinality labels and `/metrics` self-traffic are excluded to keep telemetry lightweight."

### Step 8: GitHub Actions CI/CD (30 secs)
* **Action:** Open `https://github.com/Anmol-tech17/VisionQC/actions` in browser.
* **Talking Point:** "Every push to `main` triggers automated CI testing (import validation, config schema checking, and mock unit tests) and a Docker build workflow."

---

## 5. Essential Commands Reference

### Service Management (Ubuntu-D WSL)
```bash
# Check running containers
docker ps

# View live API logs
docker compose logs -f api

# View live Prometheus / Grafana logs
docker compose -f docker-compose.monitoring.yml logs -f

# Cleanly stop monitoring stack
docker compose -f docker-compose.monitoring.yml down

# Cleanly stop API stack
docker compose down

# Cleanly stop Airflow stack
docker compose -f docker-compose.airflow.yml down
```

### Automated Testing (Windows PowerShell)
```powershell
Set-Location "C:\Users\dell\OneDrive\Desktop\MLOPS CP\VisionQC"

# Run CI test suite (5 tests, mock-isolated, ~3 seconds)
& "C:\Program Files\Python313\python.exe" -m pytest tests/test_ci.py -v

# Run full API test with real YOLO inference (~15 seconds)
& "C:\Program Files\Python313\python.exe" tests/test_api.py
```

### Git Verification
```powershell
# Verify synchronization with GitHub
git fetch origin
git status
```

---

## 6. Live Demo Troubleshooting

| Issue | Root Cause | Verified Rapid Fix |
|---|---|---|
| **Port 8000 already in use** | A previous Python or container process is still bound to port 8000. | In PowerShell: `Get-Process python* \| Stop-Process -Force` or run FastAPI on another port: `& "C:\Program Files\Python313\python.exe" -m uvicorn src.visionqc.api.main:app --port 8001`. |
| **WSL throws `E_UNEXPECTED`** | Windows WSL2 VM state needs a reset. | In PowerShell: `wsl --shutdown`, wait 5 seconds, then restart with `wsl.exe -d Ubuntu-D -u root -e docker ps`. |
| **Model not loaded (`status: degraded`)** | `mlflow.db` path or fallback weights unresolvable. | The API automatically falls back to `artifacts/models/pcb_yolov8n_640/weights/best.pt`. Verify the file exists. |
| **Grafana shows "No Data"** | Scraper hasn't collected samples yet or API is idle. | Generate traffic by submitting 2–3 images in the GUI or running: `1..5 \| ForEach-Object { Invoke-RestMethod http://localhost:8000/health }`. |
| **Prometheus target is DOWN** | Docker container cannot reach `visionqc-api:8000`. | Confirm both containers are on `visionqc-net`: `docker network inspect visionqc-net`. If running API on host Windows, access Prometheus via host network. |
| **Airflow UI not loading** | Airflow container takes ~30–45s to initialize SQLite DB on cold boot. | Run `docker compose -f docker-compose.airflow.yml logs -f airflow` to monitor startup progress. |
