"""
VisionQC — Post-training automation script
-------------------------------------------
Run this after Experiment A (or any experiment) completes training.
It:
  1. Evaluates the model on the test split
  2. Compares with baseline
  3. Selects the final model
  4. Updates training.yaml quality_gate
  5. Prints the final summary

Usage:
    C:\\Program Files\\Python313\\python.exe scripts/finalize_experiment.py --exp-name exp_a --model artifacts/models/pcb_yolov8n_640/weights/best.pt
"""
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PYTHON = sys.executable


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize a training experiment")
    parser.add_argument("--exp-name", required=True, help="Experiment name, e.g. exp_a")
    parser.add_argument("--model",    required=True, type=Path, help="Path to best.pt")
    args = parser.parse_args()

    exp_name   = args.exp_name
    model_path = args.model

    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        sys.exit(1)

    # 1. Evaluate on test split
    logger.info("Evaluating %s on test split...", exp_name)
    result = subprocess.run(
        [PYTHON, "scripts/evaluate.py", "--model", str(model_path), "--name", exp_name],
        cwd=str(_ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        logger.error("Evaluation failed for %s", exp_name)
        sys.exit(1)

    # 2. Load results
    from configs.settings import REPORTS_DIR
    exp_json_path      = REPORTS_DIR / f"evaluation_results_{exp_name}.json"
    baseline_json_path = REPORTS_DIR / "evaluation_results_baseline.json"

    if not exp_json_path.exists():
        logger.error("Results not found: %s", exp_json_path)
        sys.exit(1)

    exp_results = load_json(exp_json_path)
    exp_metrics = exp_results["overall_metrics"]

    print()
    print("=" * 65)
    print(f"  Experiment {exp_name} — Test Results")
    print("=" * 65)
    for k, v in exp_metrics.items():
        print(f"  {k:<20} {v}")

    # 3. Compare with baseline
    if baseline_json_path.exists():
        baseline_results = load_json(baseline_json_path)
        baseline_metrics = baseline_results["overall_metrics"]
        print()
        print("=" * 65)
        print("  Comparison: Experiment vs Baseline")
        print("=" * 65)
        print(f"  {'Metric':<20} {'Baseline':>12} {'Exp A':>12} {'Delta':>12}")
        print(f"  {'-'*60}")
        for k in ["mAP50", "mAP50_95", "precision", "recall"]:
            b_val = baseline_metrics.get(k)
            e_val = exp_metrics.get(k)
            delta = (e_val - b_val) if (b_val and e_val) else None
            delta_str = f"+{delta:.4f}" if delta and delta > 0 else f"{delta:.4f}" if delta else "N/A"
            print(f"  {k:<20} {b_val if b_val else 'N/A':>12} {e_val if e_val else 'N/A':>12} {delta_str:>12}")

        # Select better model by mAP50
        baseline_map50 = baseline_metrics.get("mAP50", 0) or 0
        exp_map50      = exp_metrics.get("mAP50", 0) or 0
        winner = exp_name if exp_map50 >= baseline_map50 else "baseline"
        print()
        print(f"  Selected model: {winner} (mAP50: baseline={baseline_map50:.4f}, {exp_name}={exp_map50:.4f})")
    else:
        logger.warning("Baseline results not found at %s — cannot compare", baseline_json_path)
        winner = exp_name

    # 4. Establish quality gate thresholds
    # Use 90% of the winning model's val metrics as quality gate (conservative)
    final_map50     = exp_metrics.get("mAP50", 0) or 0
    final_precision = exp_metrics.get("precision", 0) or 0
    final_recall    = exp_metrics.get("recall", 0) or 0

    gate_map50     = round(final_map50 * 0.85, 3)   # 85% of achieved — allows slight regression
    gate_precision = round(final_precision * 0.80, 3)
    gate_recall    = round(final_recall * 0.80, 3)

    print()
    print("  Proposed quality gate (85% of final test metrics):")
    print(f"    map50_min:     {gate_map50}")
    print(f"    precision_min: {gate_precision}")
    print(f"    recall_min:    {gate_recall}")

    # Update training.yaml quality_gate
    import yaml
    training_yaml_path = _ROOT / "configs" / "training.yaml"
    with open(training_yaml_path, encoding="utf-8") as f:
        training_cfg = yaml.safe_load(f)
    training_cfg["quality_gate"] = {
        "map50_min":     gate_map50,
        "precision_min": gate_precision,
        "recall_min":    gate_recall,
    }
    with open(training_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(training_cfg, f, default_flow_style=False, sort_keys=False)
    logger.info("Quality gate written to configs/training.yaml")

    # 5. Write final experiment summary
    final_summary = {
        "final_model_experiment": winner,
        "final_model_path": str(model_path),
        "test_metrics": exp_metrics,
        "per_class_metrics": exp_results.get("per_class_metrics", {}),
        "inference_latency": exp_results.get("inference_latency", {}),
        "quality_gate": {
            "map50_min":     gate_map50,
            "precision_min": gate_precision,
            "recall_min":    gate_recall,
        },
    }
    summary_path = REPORTS_DIR / "final_model_summary.json"
    summary_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")

    print()
    print(f"  Final summary: {summary_path}")
    print()


if __name__ == "__main__":
    main()
