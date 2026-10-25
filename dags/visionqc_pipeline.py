"""
VisionQC — Airflow DAG (Phase 5B)
===================================
Orchestrates the VisionQC training pipeline on Apache Airflow.

Pipeline stages:
  1. validate_inputs       — Check config, dataset, and annotations exist
  2. prepare_dataset       — Verify or regenerate YOLO-format dataset
  3. train_candidate       — Train a candidate model (smoke-test mode by default)
  4. evaluate_candidate    — Evaluate model on the test split
  5. compare_with_baseline — Compare candidate vs baseline using recorded metrics
  6. record_results        — Log outcome to MLflow and write a summary report

Environment variables:
  VISIONQC_PROJECT_DIR    — absolute path to the VisionQC project root
  VISIONQC_SMOKE_TEST     — if "true", use --epochs 1 --batch-size 2 (default: true)
  VISIONQC_TRAIN_CONFIG   — path to training config yaml (default: configs/training_exp_1.yaml)
  VISIONQC_EVAL_NAME      — experiment name tag for evaluate.py (default: airflow_candidate)

DAG schedule: @weekly (not triggered automatically at parse time)
Max active runs: 1 (prevents concurrent expensive training jobs)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(
    os.environ.get(
        "VISIONQC_PROJECT_DIR",
        # Inside the Airflow container, project is bind-mounted here:
        "/visionqc",
    )
)

SMOKE_TEST: bool = os.environ.get("VISIONQC_SMOKE_TEST", "true").lower() in ("1", "true", "yes")
TRAIN_CONFIG: str = os.environ.get("VISIONQC_TRAIN_CONFIG", "configs/training_exp_1.yaml")
EVAL_NAME: str = os.environ.get("VISIONQC_EVAL_NAME", "airflow_candidate")

PYTHON_BIN = sys.executable  # uses Airflow's own Python

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default DAG arguments
# ---------------------------------------------------------------------------

default_args = {
    "owner": "visionqc",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# ---------------------------------------------------------------------------
# Task implementations
# ---------------------------------------------------------------------------


def validate_inputs(**context):
    """
    Stage 1: Validate that all required inputs and configurations exist
    before spending CPU time on training.
    """
    log.info("=== Stage 1: Validate Inputs ===")
    errors = []

    # Check project directory
    if not PROJECT_DIR.exists():
        errors.append(f"Project directory not found: {PROJECT_DIR}")
    else:
        log.info("Project directory OK: %s", PROJECT_DIR)

    # Check training config
    config_path = PROJECT_DIR / TRAIN_CONFIG
    if not config_path.exists():
        errors.append(f"Training config not found: {config_path}")
    else:
        log.info("Training config OK: %s", config_path)
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        log.info("Config contents: %s", cfg)

    # Check dataset.yaml (YOLO-format prepared dataset)
    dataset_yaml = PROJECT_DIR / "data" / "prepared" / "dataset.yaml"
    if not dataset_yaml.exists():
        log.warning("dataset.yaml not found at %s — prepare_dataset task will regenerate it.", dataset_yaml)
        context["task_instance"].xcom_push(key="dataset_ready", value=False)
    else:
        log.info("dataset.yaml OK: %s", dataset_yaml)
        context["task_instance"].xcom_push(key="dataset_ready", value=True)

    # Check baseline evaluation results (for comparison step)
    baseline_json = PROJECT_DIR / "reports" / "evaluation_results_baseline.json"
    if not baseline_json.exists():
        log.warning("Baseline evaluation results not found at %s — comparison step will be skipped.", baseline_json)
        context["task_instance"].xcom_push(key="baseline_available", value=False)
    else:
        with open(baseline_json) as f:
            baseline = json.load(f)
        baseline_map50 = baseline.get("overall_metrics", {}).get("mAP50", "N/A")
        log.info("Baseline mAP50 = %s (from %s)", baseline_map50, baseline_json)
        context["task_instance"].xcom_push(key="baseline_available", value=True)
        context["task_instance"].xcom_push(key="baseline_map50", value=baseline_map50)

    if errors:
        raise RuntimeError("Validation failed:\n" + "\n".join(errors))

    log.info("All validation checks passed.")


def prepare_dataset(**context):
    """
    Stage 2: Verify the YOLO-format dataset exists.
    If dataset.yaml is missing, attempt to regenerate it.
    NOTE: Raw dataset (Rashid 2025) must be present on the host.
    """
    log.info("=== Stage 2: Prepare Dataset ===")

    dataset_ready = context["task_instance"].xcom_pull(
        task_ids="validate_inputs", key="dataset_ready"
    )

    dataset_yaml = PROJECT_DIR / "data" / "prepared" / "dataset.yaml"

    if dataset_ready and dataset_yaml.exists():
        log.info("Dataset already prepared at %s. Skipping regeneration.", dataset_yaml)
        return

    log.info("Attempting to run prepare_dataset.py ...")
    result = subprocess.run(
        [PYTHON_BIN, str(PROJECT_DIR / "scripts" / "prepare_dataset.py")],
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error("prepare_dataset.py failed:\n%s\n%s", result.stdout, result.stderr)
        raise RuntimeError(
            "Dataset preparation failed. Ensure the raw Rashid dataset is mounted at "
            f"{PROJECT_DIR}/data/raw/rashid. See logs for details."
        )

    if not dataset_yaml.exists():
        raise RuntimeError(f"prepare_dataset.py succeeded but {dataset_yaml} was not created.")

    log.info("Dataset prepared successfully.")


def train_candidate(**context):
    """
    Stage 3: Train a candidate model.
    In SMOKE_TEST mode (default): 1 epoch, batch 2 — fast CPU check.
    In full mode: uses the configured training YAML.
    """
    log.info("=== Stage 3: Train Candidate ===")
    log.info("Smoke-test mode: %s", SMOKE_TEST)

    cmd = [
        PYTHON_BIN,
        str(PROJECT_DIR / "scripts" / "train.py"),
        "--config", str(PROJECT_DIR / TRAIN_CONFIG),
    ]

    if SMOKE_TEST:
        # Override to minimal settings for quick pipeline validation
        cmd += ["--epochs", "1", "--batch-size", "2"]
        log.info("SMOKE TEST: running 1 epoch only — not a production training run.")
    else:
        log.warning("Full training mode — this may take 1–2 hours on CPU.")

    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_DIR),
        capture_output=False,  # allow logs to stream through Airflow
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"train.py exited with code {result.returncode}")

    log.info("Training stage complete.")


def evaluate_candidate(**context):
    """
    Stage 4: Evaluate the trained candidate on the test split.
    Uses scripts/evaluate.py and logs metrics to MLflow.
    """
    log.info("=== Stage 4: Evaluate Candidate ===")

    # Determine model path from the config
    import yaml
    config_path = PROJECT_DIR / TRAIN_CONFIG
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    output_dir = cfg.get("output_dir", "artifacts/models/pcb_yolov8n")
    model_path = PROJECT_DIR / output_dir / "weights" / "best.pt"

    if not model_path.exists():
        raise RuntimeError(
            f"Trained model not found at {model_path}. "
            "Was the train_candidate task successful?"
        )

    log.info("Evaluating model: %s", model_path)

    cmd = [
        PYTHON_BIN,
        str(PROJECT_DIR / "scripts" / "evaluate.py"),
        "--model", str(model_path),
        "--name", EVAL_NAME,
    ]

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_DIR),
        capture_output=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"evaluate.py exited with code {result.returncode}")

    # Read the evaluation results for downstream tasks
    eval_json = PROJECT_DIR / "reports" / f"evaluation_results_{EVAL_NAME}.json"
    if eval_json.exists():
        with open(eval_json) as f:
            eval_results = json.load(f)
        metrics = eval_results.get("overall_metrics", {})
        log.info("Candidate metrics: %s", metrics)
        context["task_instance"].xcom_push(key="candidate_metrics", value=metrics)
        context["task_instance"].xcom_push(key="candidate_map50", value=metrics.get("mAP50"))
    else:
        log.warning("Evaluation JSON not found at %s.", eval_json)


def compare_with_baseline(**context):
    """
    Stage 5: Compare candidate model metrics against the established baseline.
    Selection criterion: mAP50 on the held-out test split.
    The baseline is NOT retrained; only its stored evaluation JSON is used.
    """
    log.info("=== Stage 5: Compare with Baseline ===")

    baseline_available = context["task_instance"].xcom_pull(
        task_ids="validate_inputs", key="baseline_available"
    )

    candidate_map50 = context["task_instance"].xcom_pull(
        task_ids="evaluate_candidate", key="candidate_map50"
    )

    if not baseline_available:
        log.warning(
            "Baseline metrics not available. "
            "Cannot compare — reporting candidate metrics only."
        )
        context["task_instance"].xcom_push(key="winner", value="candidate (no baseline)")
        return

    baseline_map50 = context["task_instance"].xcom_pull(
        task_ids="validate_inputs", key="baseline_map50"
    )

    log.info("Baseline mAP50:  %.4f", baseline_map50 or 0)
    log.info("Candidate mAP50: %.4f", candidate_map50 or 0)

    if candidate_map50 is not None and baseline_map50 is not None:
        delta = candidate_map50 - baseline_map50
        if delta > 0:
            winner = "candidate"
            log.info("CANDIDATE WINS (+%.4f). Candidate is better.", delta)
        elif delta < 0:
            winner = "baseline"
            log.info("BASELINE WINS (%.4f). Candidate did not improve.", delta)
        else:
            winner = "tied"
            log.info("TIED. Candidate equals baseline.")
    else:
        winner = "unknown"
        log.warning("Could not determine winner — metrics missing.")

    context["task_instance"].xcom_push(key="winner", value=winner)
    context["task_instance"].xcom_push(
        key="comparison",
        value={
            "baseline_map50": baseline_map50,
            "candidate_map50": candidate_map50,
            "winner": winner,
        },
    )


def record_results(**context):
    """
    Stage 6: Write a pipeline run summary to reports/airflow_pipeline_summary.json.
    MLflow metrics are already logged by evaluate.py in the evaluate_candidate step.
    """
    log.info("=== Stage 6: Record Results ===")

    winner = context["task_instance"].xcom_pull(
        task_ids="compare_with_baseline", key="winner"
    )
    comparison = context["task_instance"].xcom_pull(
        task_ids="compare_with_baseline", key="comparison"
    )
    candidate_metrics = context["task_instance"].xcom_pull(
        task_ids="evaluate_candidate", key="candidate_metrics"
    )

    summary = {
        "dag_run_id": context["dag_run"].run_id,
        "execution_date": str(context["execution_date"]),
        "smoke_test_mode": SMOKE_TEST,
        "train_config": TRAIN_CONFIG,
        "eval_name": EVAL_NAME,
        "candidate_metrics": candidate_metrics,
        "comparison": comparison,
        "winner": winner,
    }

    reports_dir = PROJECT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "airflow_pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("Pipeline summary written to: %s", summary_path)
    log.info("Winner: %s", winner)
    log.info("Pipeline run complete.")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="visionqc_training_pipeline",
    default_args=default_args,
    description="VisionQC: validate → prepare → train → evaluate → compare → record",
    schedule="@weekly",          # not triggered automatically at startup
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    tags=["visionqc", "mlops", "training"],
) as dag:

    t1 = PythonOperator(
        task_id="validate_inputs",
        python_callable=validate_inputs,
    )

    t2 = PythonOperator(
        task_id="prepare_dataset",
        python_callable=prepare_dataset,
    )

    t3 = PythonOperator(
        task_id="train_candidate",
        python_callable=train_candidate,
        execution_timeout=timedelta(hours=3),  # safety cap for full training
    )

    t4 = PythonOperator(
        task_id="evaluate_candidate",
        python_callable=evaluate_candidate,
        execution_timeout=timedelta(hours=1),
    )

    t5 = PythonOperator(
        task_id="compare_with_baseline",
        python_callable=compare_with_baseline,
    )

    t6 = PythonOperator(
        task_id="record_results",
        python_callable=record_results,
    )

    # Pipeline dependency chain
    t1 >> t2 >> t3 >> t4 >> t5 >> t6
