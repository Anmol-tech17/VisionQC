"""
VisionQC — Model Evaluation Script (Phase 2)
----------------------------------------------
Evaluates a trained model on the test split.

Usage
-----
    # Evaluate the default model (artifacts/models/pcb_yolov8n/weights/best.pt)
    C:\\Program Files\\Python313\\python.exe scripts/evaluate.py

    # Evaluate a specific model
    C:\\Program Files\\Python313\\python.exe scripts/evaluate.py --model artifacts/models/pcb_yolov8n_640/weights/best.pt --name exp_a

Output
------
    reports/evaluation_results_<name>.json   — full test metrics
    reports/evaluation_results_<name>.md     — human-readable report
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def measure_inference_latency(model, images_dir: Path, n_samples: int = 20) -> dict:
    import random
    import numpy as np

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    if not image_files:
        logger.warning("No images in %s for latency measurement.", images_dir)
        return {}

    sample = random.sample(image_files, min(n_samples, len(image_files)))
    latencies_ms: list[float] = []

    for img_path in sample:
        t0 = time.perf_counter()
        model.predict(str(img_path), verbose=False)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

    arr = np.array(latencies_ms)
    return {
        "samples_measured": len(latencies_ms),
        "mean_ms":  round(float(arr.mean()), 1),
        "min_ms":   round(float(arr.min()), 1),
        "max_ms":   round(float(arr.max()), 1),
        "p95_ms":   round(float(np.percentile(arr, 95)), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VisionQC model on test split")
    parser.add_argument("--model", type=Path, default=None, help="Path to best.pt")
    parser.add_argument("--name",  type=str,  default="baseline", help="Experiment name suffix for output files")
    args = parser.parse_args()

    from configs.settings import (
        MODEL_PATH, DATASET_YAML, REPORTS_DIR, YOLO_CLASS_NAMES, ARTIFACTS_DIR,
    )

    model_path = args.model or MODEL_PATH
    if not model_path.exists():
        alt = ARTIFACTS_DIR / "weights" / "best.pt"
        if alt.exists():
            model_path = alt
        else:
            logger.error("Model not found at: %s\nRun scripts/train.py first.", model_path)
            sys.exit(1)

    if not DATASET_YAML.exists():
        logger.error("dataset.yaml not found. Run prepare_dataset.py first.")
        sys.exit(1)

    logger.info("Loading model from: %s", model_path)
    from ultralytics import YOLO
    model = YOLO(str(model_path))

    name = args.name

    print()
    print("=" * 65)
    print(f"  VisionQC — Test Set Evaluation [{name}]")
    print("=" * 65)
    print(f"  Model:    {model_path}")
    print(f"  Dataset:  {DATASET_YAML}")
    print()

    # 1. Evaluate on test split
    t_start = time.time()
    test_results = model.val(
        data=str(DATASET_YAML),
        split="test",
        verbose=True,
        plots=False,
    )
    t_eval = time.time() - t_start

    # 2. Extract overall metrics
    rd  = test_results.results_dict
    box = test_results.box

    precision = rd.get("metrics/precision(B)")
    recall    = rd.get("metrics/recall(B)")
    map50     = rd.get("metrics/mAP50(B)")
    map50_95  = rd.get("metrics/mAP50-95(B)")

    # 3. Per-class metrics
    per_class: dict[str, dict] = {}
    try:
        ap_class  = box.ap_class_index.tolist() if hasattr(box, "ap_class_index") else []
        ap50_list = box.ap50.tolist()            if hasattr(box, "ap50")           else []
        p_list    = box.p.tolist()               if hasattr(box, "p")              else []
        r_list    = box.r.tolist()               if hasattr(box, "r")              else []

        for i, cls_idx in enumerate(ap_class):
            cls_name = YOLO_CLASS_NAMES[cls_idx] if cls_idx < len(YOLO_CLASS_NAMES) else f"class_{cls_idx}"
            per_class[cls_name] = {
                "precision": round(p_list[i],    4) if i < len(p_list)    else None,
                "recall":    round(r_list[i],    4) if i < len(r_list)    else None,
                "ap50":      round(ap50_list[i], 4) if i < len(ap50_list) else None,
            }
    except Exception as exc:
        logger.warning("Per-class extraction failed: %s", exc)

    # 4. Inference latency
    test_images_dir = DATASET_YAML.parent / "test" / "images"
    logger.info("Measuring latency on test images (%d samples)...", min(20, 35))
    latency = measure_inference_latency(model, test_images_dir, n_samples=min(20, 35))

    # 5. Build results dict
    results_dict = {
        "experiment":             name,
        "model_path":             str(model_path),
        "dataset_yaml":           str(DATASET_YAML),
        "evaluation_split":       "test",
        "evaluation_duration_s":  round(t_eval, 1),
        "overall_metrics": {
            "precision": round(precision, 4) if precision is not None else None,
            "recall":    round(recall,    4) if recall    is not None else None,
            "mAP50":     round(map50,     4) if map50     is not None else None,
            "mAP50_95":  round(map50_95,  4) if map50_95  is not None else None,
        },
        "per_class_metrics": per_class,
        "inference_latency":  latency,
        "note": "All metrics from actual model inference on the held-out test split.",
    }

    # 6. Save JSON
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / f"evaluation_results_{name}.json"
    json_path.write_text(json.dumps(results_dict, indent=2, default=str), encoding="utf-8")

    # Also write the canonical evaluation_results.json
    (REPORTS_DIR / "evaluation_results.json").write_text(
        json.dumps(results_dict, indent=2, default=str), encoding="utf-8"
    )

    # 7. Markdown report
    def _fmt(v, n=4):
        return f"{v:.{n}f}" if v is not None else "N/A"

    md = [
        f"# VisionQC — Evaluation: {name}",
        "",
        f"**Model:** `{model_path}`",
        f"**Split:** test",
        "",
        "## Overall Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision | {_fmt(precision)} |",
        f"| Recall | {_fmt(recall)} |",
        f"| mAP@0.5 | {_fmt(map50)} |",
        f"| mAP@0.5:0.95 | {_fmt(map50_95)} |",
        "",
        "## Per-Class Metrics",
        "",
        "| Class | Precision | Recall | AP@0.5 |",
        "|---|---|---|---|",
    ]
    for cls_name in YOLO_CLASS_NAMES:
        m = per_class.get(cls_name, {})
        md.append(f"| {cls_name} | {_fmt(m.get('precision'))} | {_fmt(m.get('recall'))} | {_fmt(m.get('ap50'))} |")

    if latency:
        md += [
            "",
            "## Inference Latency (CPU)",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Samples | {latency.get('samples_measured')} |",
            f"| Mean | {latency.get('mean_ms')} ms |",
            f"| Min | {latency.get('min_ms')} ms |",
            f"| Max | {latency.get('max_ms')} ms |",
            f"| P95 | {latency.get('p95_ms')} ms |",
        ]

    md.append("")
    md.append("> All values from actual inference on held-out test split.")

    md_path = REPORTS_DIR / f"evaluation_results_{name}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    (REPORTS_DIR / "evaluation_results.md").write_text("\n".join(md), encoding="utf-8")

    # 8. Print
    print()
    print("=" * 65)
    print(f"  EVALUATION RESULTS — {name} (test split)")
    print("=" * 65)
    om = results_dict["overall_metrics"]
    for k, v in om.items():
        print(f"  {k:<20} {v if v is not None else 'N/A'}")
    if per_class:
        print()
        print(f"  Per-class AP@0.5:")
        for cls_name in YOLO_CLASS_NAMES:
            ap = per_class.get(cls_name, {}).get("ap50")
            bar = "#" * int((ap or 0) * 30)
            print(f"    {cls_name:<22} {_fmt(ap)}  {bar}")
    if latency:
        print()
        print(f"  Latency: mean={latency['mean_ms']}ms  P95={latency['p95_ms']}ms")
    print()
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    print()


if __name__ == "__main__":
    main()
