# VisionQC — Live Demonstration Checklist

**Target Duration:** 10–12 minutes  
**Audience:** Academic Evaluators / MLOps Course Reviewers  
**Project:** VisionQC: Automated Manufacturing Defect Inspection  

---

## 1. Quick Service Directory & Credentials

| Service | Local URL | Credentials / Notes | Purpose |
| :--- | :--- | :--- | :--- |
| **VisionQC Dashboard** | [http://localhost:8000](http://localhost:8000) | No auth required | Web GUI for PCB defect inspection |
| **API Swagger UI** | [http://localhost:8000/docs](http://localhost:8000/docs) | OpenAPI interactive | Interactive endpoint testing |
| **Prometheus** | [http://localhost:9090](http://localhost:9090) | No auth required | Time-series metrics collection |
| **Grafana** | [http://localhost:3000](http://localhost:3000) | `admin` / `admin` | Real-time observability dashboard |
| **Apache Airflow** | [http://localhost:8080](http://localhost:8080) | `admin` / `admin` | Pipeline orchestration DAG |
| **GitHub Repository** | [Anmol-tech17/VisionQC](https://github.com/Anmol-tech17/VisionQC) | Public repo | CI/CD actions & code versioning |

---

## 2. Pre-Flight Launch (2 Minutes Before Demo)

Run these commands from PowerShell or WSL inside `C:\Users\dell\OneDrive\Desktop\MLOPS CP\VisionQC`:

### Option A: Complete Docker Stack (Recommended)
```bash
# In Ubuntu-D WSL2:
cd "/mnt/c/Users/dell/OneDrive/Desktop/MLOPS CP/VisionQC"

# 1. Start API inference service
docker compose up -d

# 2. Start Prometheus & Grafana monitoring stack
docker compose -f docker-compose.monitoring.yml up -d

# 3. (Optional) Start Apache Airflow pipeline orchestrator
docker compose -f docker-compose.airflow.yml up -d
```

### Option B: Native Host Fast Launch (Zero Docker Overhead)
```powershell
# In Windows PowerShell:
cd "C:\Users\dell\OneDrive\Desktop\MLOPS CP\VisionQC"
python -m uvicorn src.visionqc.api.main:app --host 0.0.0.0 --port 8000
```

---

## 3. Live Demonstration Walkthrough (Chronological)

### Step 1: System Health & Champion Model (1 min)
1. Open browser to `http://localhost:8000/health`.
2. **Highlight to Evaluators:**
   - `"model_loaded": true`
   - `"model_version": 2`
   - `"model_name": "VisionQC-Detector v2"`
3. **Talking Point:** *"The inference service dynamically loads our Champion model (Version 2) from the MLflow Model Registry at startup without falling back to local defaults."*

---

### Step 2: Interactive Web Dashboard & Real Defect Inspection (2 min)
1. Open `http://localhost:8000`.
2. Notice the top status badge: `● Model Active: VisionQC-Detector v2`.
3. Drag & drop sample PCB test image:
   `VisionQC/data/prepared/test/images/pcb_defect_007.jpg`
4. Click **Run Defect Inspection**.
5. **Show Results:**
   - 6 detected defects (`mouse_bite`, `spur`, `spurious_copper`, `open_circuit`, `short`, `missing_pad`).
   - Annotated image with color-coded bounding boxes.
   - Inference latency display (~30s on CPU).
6. **Show Error Handling:** Upload a non-image text file; observe the clean HTTP 400 rejection toast.

---

### Step 3: Swagger UI OpenAPI Exploration (1 min)
1. Open `http://localhost:8000/docs`.
2. Demonstrate documented endpoints:
   - `GET /health`
   - `POST /predict`
   - `GET /metrics`

---

### Step 4: Prometheus & Grafana Observability (3 min)
1. Open Prometheus at `http://localhost:9090/targets`.
   - Show `visionqc-api` target in state `UP` with scrape interval 10s.
2. In Prometheus Graph tab, run query:
   ```promql
   http_requests_total
   ```
   - Show request counts for `/health`, `/docs`, `/predict` (both 200 and 400 status codes).
3. Open Grafana at `http://localhost:3000` (login: `admin` / `admin`).
4. Navigate to **Dashboards → VisionQC API Monitoring**.
5. **Highlight the 9 provisioned panels:**
   - Total API Requests & Prediction Counters
   - Response Latency percentiles (p50 / p95 / p99)
   - Real-time HTTP Status Codes breakdown
   - Model Inference Duration histogram

---

### Step 5: Airflow Training Pipeline Orchestration (2 min)
1. Open Airflow at `http://localhost:8080` (login: `admin` / `admin`).
2. Click on DAG `visionqc_training_pipeline`.
3. View **Graph View**:
   - `validate_inputs` ➔ `prepare_dataset` ➔ `train_candidate` ➔ `evaluate_candidate` ➔ `compare_with_baseline` ➔ `record_results`
4. **Talking Point:** *"Airflow automates the end-to-end retraining cycle. It validates inputs, prepares YOLO data, trains a candidate model in safe smoke-test mode (`VISIONQC_SMOKE_TEST=true`), evaluates against test data, compares against baseline mAP50 (0.7527), and records results without overwriting production champions."*

---

### Step 6: Data & Model Versioning (DVC + MLflow) (1 min)
1. In terminal or IDE, show `data/raw/rashid.dvc`.
   - Explain DVC tracks large PCB datasets via content hashes (`5543c7b...`) offloaded to external storage.
2. Show `mlflow.db` and registered model alias `champion` pointing to Version 2.

---

### Step 7: CI/CD GitHub Actions (1 min)
1. Open GitHub: [https://github.com/Anmol-tech17/VisionQC/actions](https://github.com/Anmol-tech17/VisionQC/actions).
2. Show recent green builds for:
   - `CI — Tests & Lint` (5/5 unit tests, syntax checks, YAML validation)
   - `Docker Build` (container build automation)

---

## 4. Post-Demo Clean Shutdown

When finished, run:
```bash
# In Ubuntu-D WSL2:
cd "/mnt/c/Users/dell/OneDrive/Desktop/MLOPS CP/VisionQC"
docker compose -f docker-compose.monitoring.yml down
docker compose -f docker-compose.airflow.yml down
docker compose down
```

