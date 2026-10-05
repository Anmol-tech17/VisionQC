
"""
VisionQC — Model Training Script
-----------------------------------
Trains a YOLOv8n detector on the prepared PCB defect dataset.

Usage
-----
    python scripts/train.py                         # use configs/training.yaml defaults
    python scripts/train.py --epochs 30             # override epochs
    python scripts/train.py --batch-size 8          # override batch size
    python scripts/train.py --imgsz 416             # override image size

Prerequisites
-------------
    Run scripts/prepare_dataset.py first to create data/prepared/

Output
------
    artifacts/models/pcb_yolov8n/weights/best.pt   — best checkpoint
    artifacts/models/pcb_yolov8n/weights/last.pt   — last checkpoint
    artifacts/models/pcb_yolov8n/results.csv       — per-epoch metrics
    reports/training_run.json                      — full configuration + results

All metrics come from the actual Ultralytics training loop.
"""

import argparse
import json
import logging
import random
import shutil
import sys
import time
from pathlib import Path

# -- resolve project root --
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def set_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)   # deterministic mode is too slow on CPU
    logger.info("Random seed set to: %d", seed)


def load_training_config(config_path: Path) -> dict:
    """Load training.yaml and return as a dict."""
    import yaml  # bundled with ultralytics

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Override config dict with any CLI arguments that were explicitly provided."""
    if args.epochs is not None:
        logger.info("CLI override: epochs = %d (was %s)", args.epochs, cfg.get("epochs"))
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        logger.info("CLI override: batch_size = %d (was %s)", args.batch_size, cfg.get("batch_size"))
        cfg["batch_size"] = args.batch_size
    if args.imgsz is not None:
        logger.info("CLI override: image_size = %d (was %s)", args.imgsz, cfg.get("image_size"))
        cfg["image_size"] = args.imgsz
    if args.patience is not None:
        logger.info("CLI override: patience = %d (was %s)", args.patience, cfg.get("patience"))
        cfg["patience"] = args.patience
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train VisionQC PCB defect detector (YOLOv8n)"
    )
    parser.add_argument("--epochs",     type=int,   default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int,   dest="batch_size", default=None, help="Batch size")
    parser.add_argument("--imgsz",      type=int,   default=None, help="Training image size (pixels)")
    parser.add_argument("--patience",   type=int,   default=None, help="Early stopping patience")
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "configs" / "training.yaml",
        help="Path to training.yaml",
    )
    args = parser.parse_args()

    # 1. Load config
    logger.info("Loading training config from: %s", args.config)
    cfg = load_training_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    seed       = cfg.get("seed", 42)
    model_name = cfg.get("model", "yolov8n.pt")
    epochs     = cfg.get("epochs", 20)
    batch_size = cfg.get("batch_size", 4)
    imgsz      = cfg.get("image_size", 320)
    patience   = cfg.get("patience", 5)
    device     = cfg.get("device", "cpu")
    workers    = cfg.get("workers", 0)
    output_dir = _PROJECT_ROOT / cfg.get("output_dir", "artifacts/models/pcb_yolov8n")

    from configs.settings import DATASET_YAML, REPORTS_DIR, ARTIFACTS_DIR

    if not DATASET_YAML.exists():
        logger.error(
            "dataset.yaml not found at: %s\n"
            "Run scripts/prepare_dataset.py first.",
            DATASET_YAML,
        )
        sys.exit(1)

    # 2. Set seeds
    set_seeds(seed)

    # 3. Import ultralytics
    try:
        from ultralytics import YOLO
        import ultralytics
        logger.info("Ultralytics version: %s", ultralytics.__version__)
    except ImportError as exc:
        logger.error("ultralytics is not installed: %s", exc)
        sys.exit(1)

    # 4. Print training plan
    print()
    print("=" * 62)
    print("  VisionQC — YOLOv8n Training")
    print("=" * 62)
    print(f"  Model:       {model_name}")
    print(f"  Dataset:     {DATASET_YAML}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  Image size:  {imgsz}px")
    print(f"  Patience:    {patience}")
    print(f"  Device:      {device}")
    print(f"  Workers:     {workers}")
    print(f"  Seed:        {seed}")
    print(f"  Output:      {output_dir}")
    print("=" * 62)
    print()
    print("NOTE: CPU training with 2 threads — this will take time.")
    print("      Do not interrupt. Early stopping may end sooner.")
    print()

    # 5. Train
    t_start = time.time()

    import mlflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("VisionQC_Training")
    
    # We will log the run with the experiment name in MLflow
    # Using output_dir.name as run name, usually it is something like pcb_yolov8n
    run_name = args.config.stem if args.config else output_dir.name
    
    run = mlflow.start_run(run_name=run_name)
    logger.info("Started MLflow run: %s", run.info.run_id)
    
    # Log parameters
    mlflow.log_params({
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": imgsz,
        "patience": patience,
        "device": device,
        "workers": workers,
        "seed": seed,
        "dataset_yaml": str(DATASET_YAML)
    })

    model = YOLO(model_name)   # loads pretrained COCO weights

    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        patience=patience,
        device=device,
        workers=workers,
        seed=seed,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        verbose=True,
        plots=False,        # disable plots (saves CPU time)
        save=True,
        save_period=-1,     # only save best and last
        cache=False,        # don't cache images (saves RAM)
        rect=False,         # rectangular training off (small dataset)
        cos_lr=False,       # simple LR schedule
        amp=False,          # no mixed precision on CPU
        pretrained=True,    # use COCO pretrained weights
    )

    t_elapsed = time.time() - t_start
    duration_min = t_elapsed / 60

    # 6. Copy weights to expected location
    # ultralytics saves to: output_dir/weights/best.pt
    weights_src = output_dir / "weights" / "best.pt"
    weights_dst = ARTIFACTS_DIR / "weights" / "best.pt"

    if weights_src.exists():
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS_DIR / "weights").mkdir(exist_ok=True)
        if weights_src.resolve() != weights_dst.resolve():
            shutil.copy2(weights_src, weights_dst)
        logger.info("Best model saved to: %s", weights_dst)
    else:
        logger.warning(
            "best.pt not found at expected path: %s\n"
            "Check the ultralytics output directory.",
            weights_src,
        )

    # 7. Extract validation metrics from training results
    # results.results_dict contains the final epoch metrics
    val_metrics: dict = {}
    try:
        rd = results.results_dict
        val_metrics = {
            "metrics/precision(B)":   rd.get("metrics/precision(B)", None),
            "metrics/recall(B)":      rd.get("metrics/recall(B)", None),
            "metrics/mAP50(B)":       rd.get("metrics/mAP50(B)", None),
            "metrics/mAP50-95(B)":    rd.get("metrics/mAP50-95(B)", None),
        }
    except Exception as exc:
        logger.warning("Could not read results_dict: %s", exc)

    # 8. Save training run record
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    run_record = {
        "training_config": {
            "model": model_name,
            "epochs_requested": epochs,
            "epochs_completed": getattr(results, "epoch", epochs),
            "batch_size": batch_size,
            "image_size": imgsz,
            "patience": patience,
            "device": device,
            "workers": workers,
            "seed": seed,
            "dataset_yaml": str(DATASET_YAML),
        },
        "duration_seconds": round(t_elapsed, 1),
        "duration_minutes": round(duration_min, 2),
        "best_model_path": str(weights_dst) if weights_dst.exists() else "not found",
        "validation_metrics_at_best_epoch": val_metrics,
        "note": (
            "Metrics are from ultralytics validation on the val split at the best epoch. "
            "Run scripts/evaluate.py for test-split metrics."
        ),
    }

    run_record_path = REPORTS_DIR / "training_run.json"
    run_record_path.write_text(
        json.dumps(run_record, indent=2, default=str), encoding="utf-8"
    )

    # 8.5 MLflow Log Metrics and Artifacts
    try:
        import mlflow
        if mlflow.active_run():
            # Log metrics
            if val_metrics:
                mlflow.log_metrics({
                    "val_precision": val_metrics.get("metrics/precision(B)") or 0,
                    "val_recall": val_metrics.get("metrics/recall(B)") or 0,
                    "val_mAP50": val_metrics.get("metrics/mAP50(B)") or 0,
                    "val_mAP50_95": val_metrics.get("metrics/mAP50-95(B)") or 0,
                })
            
            mlflow.log_metric("duration_seconds", t_elapsed)
            
            # Log artifacts
            if weights_dst.exists():
                mlflow.log_artifact(str(weights_dst), artifact_path="model_weights")
            
            if run_record_path.exists():
                mlflow.log_artifact(str(run_record_path))
                
            results_csv = output_dir / "results.csv"
            if results_csv.exists():
                mlflow.log_artifact(str(results_csv))
                
            confusion_matrix = output_dir / "confusion_matrix.png"
            if confusion_matrix.exists():
                mlflow.log_artifact(str(confusion_matrix))
                
            logger.info("MLflow tracking completed.")
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)

    # 9. Print summary
    print()
    print("=" * 62)
    print("  TRAINING COMPLETE")
    print("=" * 62)
    print(f"  Duration:   {duration_min:.1f} minutes ({t_elapsed:.0f} seconds)")
    print(f"  Best model: {weights_dst}")
    if val_metrics:
        print()
        print("  Validation metrics (best epoch):")
        for k, v in val_metrics.items():
            label = k.split("/")[1].replace("(B)", "")
            val_str = f"{v:.4f}" if v is not None else "N/A"
            print(f"    {label:<22} {val_str}")
    print()
    print(f"  Full run record: {run_record_path}")
    print()
    print()
    print("  Next step: python scripts/evaluate.py")
    print()

    # Close MLflow run
    import mlflow
    mlflow.end_run()


if __name__ == "__main__":
    main()
