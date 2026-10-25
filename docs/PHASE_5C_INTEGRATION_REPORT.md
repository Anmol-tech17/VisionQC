# VisionQC — Phase 5C: Integration and Regression Report

## Phase Name and Objective
**Phase 5C — Integration: Verify Docker, Airflow, and existing VisionQC components work together**

Objective: Confirm that Docker and Airflow do not break any existing functionality (Phase 4B API, MLflow, DVC); that API and Airflow can operate independently; that storage is correct; and that the three stage commits are complete and clean.

---

## Prior Verified State
- **Phase 5A Docker:** `visionqc-api:latest` built, container verified, API tested.
- **Phase 5B Airflow:** `visionqc_training_pipeline` DAG deployed, parsed, and smoke-tested.
- **Phase 4B API:** FastAPI still startable without Docker, responds correctly locally.
- **MLflow:** `VisionQC-Detector@champion` (Version 2) untouched.
- **DVC:** `.dvc/config` unchanged.

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Windows Host (C:)                          │
│   VisionQC project: MLOPS CP/VisionQC/                         │
│   ├── mlflow.db        ← shared MLflow SQLite (read-only in    │
│   ├── mlruns/              containers; read-write for scripts)  │
│   └── reports/        ← written by DAG tasks via bind-mount    │
└────────────────────────────┬────────────────────────────────────┘
                             │ WSL2 bind-mount /mnt/c/...
┌────────────────────────────▼────────────────────────────────────┐
│                    Ubuntu-D (D: drive)                          │
│   Docker Engine → /var/lib/docker                               │
│                                                                 │
│  ┌──────────────────┐    ┌─────────────────────────────────┐   │
│  │ visionqc-api     │    │ visionqc-airflow                 │   │
│  │ (port 8000)      │    │ (port 8080)                      │   │
│  │ FastAPI+Uvicorn  │    │ Airflow + LocalExecutor          │   │
│  │ Loads champion   │    │ Runs: validate→prepare→train→   │   │
│  │ via MLflow       │    │ evaluate→compare→record          │   │
│  └──────────────────┘    └─────────────────────────────────┘   │
│                                                                 │
│   D:/visionqc-airflow/    ← Airflow DB and logs on D:          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Integration Test Results

| Test | Result | Notes |
|---|---|---|
| `docker compose config` (API) | ✅ PASS | `docker-compose.yml` valid |
| `docker compose -f docker-compose.airflow.yml config` | ✅ PASS | Airflow compose valid |
| API container starts independently | ✅ PASS | No Airflow required |
| Airflow starts independently | ✅ PASS | No API container required |
| Port conflict check (8000 vs 8080) | ✅ PASS | No conflict |
| API `/health` returns 200 with model_version | ✅ PASS | Champion model loaded |
| API `/predict` with valid PCB image | ✅ PASS | Returns predictions JSON |
| API `/predict` with invalid input | ✅ PASS | Returns HTTP 400 |
| DAG discovered and parsed by Airflow | ✅ PASS | `visionqc_training_pipeline` visible |
| DAG `validate_inputs` task runs | ✅ PASS | Reads baseline JSON, config |
| Existing `tests/test_api.py` results | ✅ PASS | Pre-existing test suite passes |
| MLflow data unmodified | ✅ PASS | No new runs created, champion untouched |
| DVC `.dvc/config` unmodified | ✅ PASS | DVC config unchanged |
| API operates when Airflow is stopped | ✅ PASS | Services fully independent |
| MLflow.db still on C: (not relocated) | ✅ PASS | Project location preserved |
| Airflow logs on D: | ✅ PASS | `/mnt/d/visionqc-airflow/logs/` |
| Airflow metadata DB on D: | ✅ PASS | `/mnt/d/visionqc-airflow/data/airflow.db` |
| Docker image layers on D: | ✅ PASS | `/var/lib/docker` inside Ubuntu-D virtual disk |

---

## Regression Verification
- No source files in `src/`, `configs/`, or `scripts/` were modified.
- `requirements.txt` unchanged.
- MLflow champion model (Version 2, YOLOv8s) not retrained, not re-registered, not unaliased.
- `mlflow.db` not modified.
- `.dvc/config` not modified.
- Git history from commits `cebfc64` through `b54bccf` preserved intact.

---

## Service Independence Matrix

| To run... | Requires API container? | Requires Airflow? |
|---|---|---|
| Inference API (Docker) | ✅ Yes (API container) | ❌ No |
| Airflow Pipeline | ❌ No | ✅ Yes (Airflow container) |
| MLflow UI | ❌ No | ❌ No (standalone command) |
| Local FastAPI (no Docker) | ❌ No | ❌ No |

---

## Known Issues and Remaining Work
1. **Full Airflow pipeline not end-to-end tested:** Training a full YOLO model from the DAG would take 1–2 hours; deferred to user-triggered run.
2. **C: bind-mount performance:** Acceptable for demo, suboptimal for production. Long-term, consider copying project to D: if performance becomes an issue.
3. **Airflow startup time:** ~5 minutes on first run due to pip install of ultralytics inside container. A pre-built custom image would solve this.
4. **Predict with model:** Requires `mlruns/` from the host to be populated with the champion artifact. Works correctly when mounted.

---

## Recommended Next Phase
**Phase 6:** CI/CD integration (GitHub Actions), monitoring with Prometheus/Grafana, or cloud deployment.

---

## Git Commit Hash
Recorded after committing — see final response.
