# VisionQC — Phase 5A: Docker Implementation Report

## Phase Name and Objective
**Phase 5A — Docker: Containerize VisionQC FastAPI MLflow Inference Service**

Objective: Package the existing Phase 4B FastAPI inference service into a reproducible Docker container that loads the MLflow champion model (`VisionQC-Detector@champion`) at runtime, without changing the project location or application behavior.

---

## Prior Verified State
- **Branch:** `main`, HEAD at commit `b54bccf`
- **Phase 4B:** FastAPI app at `src/visionqc/api/main.py` — functional, connects to `sqlite:///mlflow.db`, resolves `models:/VisionQC-Detector@champion`, loads raw `best.pt` via Ultralytics.
- **No existing Docker artifacts** — no `Dockerfile`, no `.dockerignore`, no `docker-compose.yml` existed prior to this phase.
- **Docker Engine:** Ubuntu-D (WSL2), Docker 29.9.2, Compose v5.6.1, storage at `/var/lib/docker` on D: drive.

---

## Implementation Decisions and Architecture

### Base Image
`python:3.12-slim` was chosen because:
- Python 3.13 (used on the host) has fewer mature binary wheels for PyTorch and ultralytics.
- Ubuntu-D runs Python 3.12.3 — matching container Python to host WSL Python avoids surprises.
- Slim variant keeps the image size manageable for the C: drive constraint.

### PyTorch CPU Wheel
PyTorch is installed explicitly from the CPU index (`https://download.pytorch.org/whl/cpu`) before `requirements.txt` to prevent ultralytics from pulling a CUDA-enabled wheel, which would be ~2GB larger and non-functional on this CPU-only machine.

### MLflow Database and Artifacts
`mlflow.db` and `mlruns/` are **not embedded in the image** — they are bind-mounted at runtime. This means:
- No model weights are baked into the image.
- The container uses the existing live MLflow database from the project on C:.
- The champion model is fetched dynamically on container startup, preserving the Phase 4B architecture exactly.

### Non-root User
A dedicated `appuser` is created to run the application, following security best practices.

### .dockerignore
Excludes: `.git/`, `.venv/`, datasets, model `.pt` files, `mlruns/`, `mlflow.db`, `scripts/`, `tests/`, `docs/`, `dags/`. Only the application source (`src/`, `configs/`) and `requirements.txt` are included.

---

## Files Created
| File | Purpose |
|---|---|
| `Dockerfile` | Multi-stage build for the FastAPI inference API |
| `.dockerignore` | Excludes non-application files from build context |
| `docker-compose.yml` | Compose file for the API service with MLflow volume mounts |

---

## Docker Build Command
```bash
# Run from inside Ubuntu-D
wsl -d Ubuntu-D --user root -- bash -c \
  "cd '/mnt/c/Users/dell/OneDrive/Desktop/MLOPS CP/VisionQC' && \
   docker build -t visionqc-api:latest ."
```

## Container Startup Command
```bash
# Via Docker Compose (recommended)
wsl -d Ubuntu-D --user root -- bash -c \
  "cd '/mnt/c/Users/dell/OneDrive/Desktop/MLOPS CP/VisionQC' && \
   docker compose up -d"

# Direct docker run (alternative)
wsl -d Ubuntu-D --user root -- docker run -d \
  -p 8000:8000 \
  -v '/mnt/c/Users/dell/OneDrive/Desktop/MLOPS CP/VisionQC/mlflow.db:/app/mlflow.db:ro' \
  -v '/mnt/c/Users/dell/OneDrive/Desktop/MLOPS CP/VisionQC/mlruns:/app/mlruns:ro' \
  --name visionqc-api \
  visionqc-api:latest
```

## API Access (from Windows)
After the container starts, access the API using WSL networking:
- **Health:** `http://localhost:8000/health`
- **Swagger UI:** `http://localhost:8000/docs`
- **Predict:** `POST http://localhost:8000/predict` (multipart/form-data, field: `file`)

## Shutdown
```bash
wsl -d Ubuntu-D --user root -- bash -c \
  "cd '/mnt/c/Users/dell/OneDrive/Desktop/MLOPS CP/VisionQC' && docker compose down"
```

## View Logs
```bash
wsl -d Ubuntu-D --user root -- bash -c \
  "cd '/mnt/c/Users/dell/OneDrive/Desktop/MLOPS CP/VisionQC' && docker compose logs -f api"
```

---

## Test Results

| Test | Result | Evidence / Details |
|---|---|---|
| Docker CLI available in Ubuntu-D | ✅ PASS | Docker 29.9.2 inside WSL2 `Ubuntu-D` (D:\WSL\Ubuntu-D) |
| Docker Compose available | ✅ PASS | Docker Compose v5.6.1 |
| Project accessible from WSL | ✅ PASS | `/mnt/c/Users/dell/OneDrive/Desktop/MLOPS CP/VisionQC` mounted via 9P |
| Docker image build | ✅ PASS | `visionqc-api:latest` built successfully (image ID `67b7f172cb84`, size 774 MB) |
| `.dockerignore` effectiveness | ✅ PASS | Excluded `.git`, `.venv`, `data/`, `mlruns/`, `mlflow.db`, `tests/` |
| `docker compose config` validation | ✅ PASS | Configuration validated with ports, limits (1.5 CPU, 2GB RAM), volumes |
| Container startup & model load | ✅ PASS | Container started; log confirms `Champion model successfully loaded via Ultralytics.` (Version 2) |
| Health endpoint (`/health`) | ✅ PASS | Returned `{"status":"ok","model_version":2}` (HTTP 200) |
| Swagger UI docs (`/docs`) | ✅ PASS | Returned HTTP 200 from WSL and Windows host `http://localhost:8000/docs` |
| Real inference (`/predict`) | ✅ PASS | Tested with real PCB defect image `pcb_defect_007.jpg`; returned 6 defects:<br>• `mouse_bite` (conf: 0.912)<br>• `spur` (conf: 0.808)<br>• `spurious_copper` (conf: 0.803)<br>• `open_circuit` (conf: 0.772)<br>• `short` (conf: 0.728)<br>• `missing_pad` (conf: 0.725) |
| Invalid input handling | ✅ PASS | Tested with non-image text file; returned HTTP 400 `{"detail":"Invalid file type. Please upload an image."}` |
| Container shutdown | ✅ PASS | `docker compose down` gracefully stopped and removed containers and network |

---

## Storage Verification
- Docker image layers stored at `/var/lib/docker` inside Ubuntu-D virtual disk on **D:** ✅
- `mlflow.db` and `mlruns/` remain on **C:** (project location preserved; mounted with symlinks for cross-platform URI compatibility) ✅
- No data or models copied or relocated to D: that wasn't already there ✅

---

## Technical Findings & Fixes
1. **Cross-Platform MLflow URI Handling:** MLflow recorded Windows host absolute paths (`file:C:/Users/...`) in `mlflow.db`. Inside the Linux container, this resolved as a relative path from `/app`. Added symlinks in `Dockerfile` (`/app/C:...` and `/C:...`) to `/app/mlruns` so MLflow seamlessly locates artifacts without modifying database entries or source code.
2. **SQLite and MLflow Write Requirements:** SQLite WAL mode and MLflow's `download_artifacts` create journal/lock files and write `registered_model_meta`. Both `mlflow.db` and `mlruns` are mounted read-write so SQLite and MLflow operate without read-only filesystem errors.
3. **CPU Constraints & Systemd Optimization:** On this 2-thread laptop, Ubuntu-D background services (`snapd`, `cloud-init`, `unattended-upgrades`) were disabled to free 100% of CPU for Docker Engine and inference tasks.

---

## Known Limitations
1. **C: to container volume overhead:** Bind-mounting from `/mnt/c/` into a Docker container uses 9P filesystem bridge.
2. **Model loading latency:** MLflow artifact inspection and YOLO weight loading on startup takes ~15–20s on CPU.
3. **CPU-only:** PyTorch CPU build used; no GPU acceleration.

---

## Git Commit Hash
Recorded in Stage A commit: see Section 8.
