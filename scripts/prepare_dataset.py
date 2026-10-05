"""
VisionQC — Dataset Preparation CLI Script
-------------------------------------------
Runs the COCO → YOLO conversion, verifies integrity, and writes reports.

Usage:
    C:\\Program Files\\Python313\\python.exe scripts/prepare_dataset.py

Outputs:
    data/prepared/           — YOLO dataset (train/val/test)
    reports/dataset_report.md
    reports/dataset_report.json
"""
import json
import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def verify_no_leakage(stats) -> bool:
    """Verify there is no image-level leakage across splits."""
    # The converter already uses assertions, but let's double-check via filenames
    from pathlib import Path
    prepared_dir = stats.prepared_dir
    splits = {}
    for split_name in ["train", "val", "test"]:
        imgs_dir = prepared_dir / split_name / "images"
        if imgs_dir.exists():
            splits[split_name] = {f.name for f in imgs_dir.glob("*")}
        else:
            splits[split_name] = set()

    train_val = splits["train"] & splits["val"]
    train_test = splits["train"] & splits["test"]
    val_test   = splits["val"]   & splits["test"]
    ok = not (train_val or train_test or val_test)
    if not ok:
        logger.warning("Leakage detected: train∩val=%d train∩test=%d val∩test=%d",
                       len(train_val), len(train_test), len(val_test))
    return ok


def check_class_coverage(stats) -> list[str]:
    """Return list of classes with zero annotations in train split."""
    from src.visionqc.data.dataset_prep import MODEL_CLASS_NAMES
    missing = []
    train_split = next((s for s in stats.splits if s.name == "train"), None)
    if not train_split:
        return MODEL_CLASS_NAMES  # can't verify
    for cls_name in MODEL_CLASS_NAMES:
        if train_split.class_distribution.get(cls_name, 0) == 0:
            missing.append(cls_name)
    return missing


def main() -> None:
    from src.visionqc.data.dataset_prep import COCOToYOLOConverter
    from configs.settings import REPORTS_DIR, ANNOTATION_FILE, IMAGES_DIR

    print()
    print("=" * 62)
    print("  VisionQC — Dataset Preparation")
    print("=" * 62)

    # Run conversion
    converter = COCOToYOLOConverter()
    stats = converter.convert()

    print()
    print(stats.summary_text())

    # Verify no leakage
    ok = verify_no_leakage(stats)
    if ok:
        print("[OK] No image leakage between splits.")
    else:
        print("[FAIL] WARNING: Possible image leakage detected!")

    # Check class coverage
    missing_classes = check_class_coverage(stats)
    if missing_classes:
        print(f"\n[WARNING] The following classes have ZERO annotations: {missing_classes}")
        print("  This will prevent the model from learning these classes.")
    else:
        print("[OK] All 6 defect classes have annotations.")

    # Save dataset report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    report_data = {
        "total_images": stats.total_images,
        "total_annotations": stats.total_annotations,
        "num_classes": stats.num_classes,
        "seed": stats.seed,
        "split_ratios": stats.split_ratios,
        "splits": [
            {
                "name": s.name,
                "image_count": s.image_count,
                "annotation_count": s.annotation_count,
                "class_distribution": s.class_distribution,
            }
            for s in stats.splits
        ],
        "warnings": stats.warnings,
        "leakage_check": "passed" if ok else "FAILED",
        "class_coverage_check": "passed" if not missing_classes else f"FAILED: {missing_classes}",
    }
    json_path = REPORTS_DIR / "dataset_report.json"
    json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")

    # Markdown
    from src.visionqc.data.dataset_prep import MODEL_CLASS_NAMES
    all_classes: dict[str, int] = {}
    for s in stats.splits:
        for cls, cnt in s.class_distribution.items():
            all_classes[cls] = all_classes.get(cls, 0) + cnt

    md_lines = [
        "# VisionQC — Dataset Preparation Report",
        "",
        f"**Total images:** {stats.total_images}  ",
        f"**Total annotations:** {stats.total_annotations}  ",
        f"**Defect classes:** {stats.num_classes}  ",
        f"**Seed:** {stats.seed}  ",
        "",
        "## Split Summary",
        "",
        "| Split | Images | Annotations |",
        "|---|---|---|",
    ]
    for s in stats.splits:
        md_lines.append(f"| {s.name} | {s.image_count} | {s.annotation_count} |")

    md_lines += [
        "",
        "## Class Distribution (All Splits)",
        "",
        "| Class | Total Instances |",
        "|---|---|",
    ]
    for cls_name in MODEL_CLASS_NAMES:
        md_lines.append(f"| {cls_name} | {all_classes.get(cls_name, 0)} |")

    md_lines += [
        "",
        f"**Leakage check:** {'PASSED' if ok else 'FAILED'}  ",
        f"**Class coverage:** {'PASSED' if not missing_classes else 'ISSUES: ' + str(missing_classes)}  ",
    ]
    if stats.warnings:
        md_lines += ["", "## Warnings", ""]
        for w in stats.warnings:
            md_lines.append(f"- {w}")

    md_path = REPORTS_DIR / "dataset_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print()
    print(f"Reports saved:")
    print(f"  JSON: {json_path}")
    print(f"  MD:   {md_path}")
    print()
    print("Next: C:\\Program Files\\Python313\\python.exe scripts/train.py")
    print()


if __name__ == "__main__":
    main()
