"""
VisionQC — COCO to YOLO Dataset Converter
-------------------------------------------
Reads the COCO annotation JSON, filters out the root category,
maps the six defect classes to contiguous 0-based model IDs,
performs a reproducible image-level train/val/test split, and
writes a YOLO-format dataset ready for ultralytics training.

Class mapping (original COCO ID → model ID)
-------------------------------------------
    COCO 1 (missing_pad)     → model 0
    COCO 2 (mouse_bite)      → model 1
    COCO 3 (open_circuit)    → model 2
    COCO 4 (short)           → model 3
    COCO 5 (spur)            → model 4
    COCO 6 (spurious_copper) → model 5

YOLO label format (per line in .txt file)
-----------------------------------------
    <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>

All values are normalised to [0, 1] relative to the image dimensions.

Usage
-----
    from src.visionqc.data.dataset_prep import COCOToYOLOConverter
    converter = COCOToYOLOConverter()
    stats = converter.convert()
    print(stats)
"""

import json
import logging
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# COCO category ID → model class ID (0-based, contiguous)
# Category 0 (detecting-pcb-defects) is the root label and is intentionally excluded.
# ---------------------------------------------------------------------------
COCO_TO_MODEL_ID: dict[int, int] = {
    1: 0,  # missing_pad
    2: 1,  # mouse_bite
    3: 2,  # open_circuit
    4: 3,  # short
    5: 4,  # spur
    6: 5,  # spurious_copper
}

MODEL_CLASS_NAMES: list[str] = [
    "missing_pad",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SplitInfo:
    name: str
    image_count: int
    annotation_count: int
    class_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class DatasetStats:
    total_images: int
    total_annotations: int
    num_classes: int
    splits: list[SplitInfo]
    seed: int
    split_ratios: dict[str, float]
    prepared_dir: Path
    dataset_yaml: Path
    warnings: list[str] = field(default_factory=list)

    def summary_text(self) -> str:
        lines = [
            "DATASET PREPARATION COMPLETE",
            f"  Total images:      {self.total_images}",
            f"  Total annotations: {self.total_annotations}",
            f"  Defect classes:    {self.num_classes}",
            f"  Seed:              {self.seed}",
            "",
            f"  {'Split':<10} {'Images':>8} {'Annotations':>13}",
            "  " + "-" * 33,
        ]
        for s in self.splits:
            lines.append(f"  {s.name:<10} {s.image_count:>8} {s.annotation_count:>13}")
        lines.append("")
        lines.append("  Class distribution across all splits:")
        all_classes: dict[str, int] = {}
        for s in self.splits:
            for cls, cnt in s.class_distribution.items():
                all_classes[cls] = all_classes.get(cls, 0) + cnt
        for cls_name in MODEL_CLASS_NAMES:
            cnt = all_classes.get(cls_name, 0)
            lines.append(f"    {cls_name:<22} {cnt:>5} instances")
        if self.warnings:
            lines.append("")
            for w in self.warnings:
                lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

class COCOToYOLOConverter:
    """
    Converts a COCO annotation file to YOLO format with an image-level split.

    Parameters
    ----------
    annotation_file : Path
        Path to _annotations.coco.json
    images_dir : Path
        Directory containing the raw PCB images
    prepared_dir : Path
        Output root for the YOLO dataset
    train_ratio : float
        Fraction of images for training (default 0.70)
    val_ratio : float
        Fraction of images for validation (default 0.15); test gets the rest
    seed : int
        Random seed for reproducible splits
    """

    def __init__(
        self,
        annotation_file: Optional[Path] = None,
        images_dir: Optional[Path] = None,
        prepared_dir: Optional[Path] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> None:
        from configs.settings import (
            ANNOTATION_FILE,
            IMAGES_DIR,
            PREPARED_DIR,
            YOLO_CLASS_NAMES,
        )

        self.annotation_file = annotation_file or ANNOTATION_FILE
        self.images_dir = images_dir or IMAGES_DIR
        self.prepared_dir = prepared_dir or PREPARED_DIR
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        self.seed = seed
        self.class_names = YOLO_CLASS_NAMES

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def convert(self) -> DatasetStats:
        """
        Run the full conversion pipeline.

        Returns
        -------
        DatasetStats
            Summary of the prepared dataset including split counts
            and class distribution per split.
        """
        warnings: list[str] = []

        # 1. Load COCO JSON
        logger.info("Loading COCO annotations from: %s", self.annotation_file)
        with open(self.annotation_file, encoding="utf-8") as f:
            coco = json.load(f)

        images = coco["images"]
        annotations = coco["annotations"]

        # 2. Build image_id → image record map
        id_to_image: dict[int, dict] = {img["id"]: img for img in images}

        # 3. Build image_id → list[annotation] index (defect classes only)
        image_annotations: dict[int, list[dict]] = {
            img["id"]: [] for img in images
        }
        skipped = 0
        for ann in annotations:
            if ann["category_id"] not in COCO_TO_MODEL_ID:
                skipped += 1
                continue
            image_annotations[ann["image_id"]].append(ann)

        if skipped:
            warnings.append(
                f"Skipped {skipped} annotations with non-defect category (e.g. category 0)."
            )

        # 4. Only include images that actually exist on disk
        valid_image_ids: list[int] = []
        missing: list[str] = []
        for img in images:
            img_path = self.images_dir / img["file_name"]
            if img_path.exists():
                valid_image_ids.append(img["id"])
            else:
                missing.append(img["file_name"])

        if missing:
            warnings.append(
                f"{len(missing)} image file(s) not found on disk and excluded."
            )

        logger.info("Valid images: %d", len(valid_image_ids))

        # 5. Reproducible image-level split
        rng = random.Random(self.seed)
        shuffled_ids = list(valid_image_ids)
        rng.shuffle(shuffled_ids)

        n_total = len(shuffled_ids)
        n_train = int(n_total * self.train_ratio)
        n_val = int(n_total * self.val_ratio)

        train_ids = set(shuffled_ids[:n_train])
        val_ids = set(shuffled_ids[n_train : n_train + n_val])
        test_ids = set(shuffled_ids[n_train + n_val :])

        # Verify no overlap
        assert not (train_ids & val_ids), "Image leakage: train ∩ val"
        assert not (train_ids & test_ids), "Image leakage: train ∩ test"
        assert not (val_ids & test_ids), "Image leakage: val ∩ test"

        splits = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }

        # 6. Write YOLO files
        self.prepared_dir.mkdir(parents=True, exist_ok=True)

        split_infos: list[SplitInfo] = []

        for split_name, id_set in splits.items():
            img_dir = self.prepared_dir / split_name / "images"
            lbl_dir = self.prepared_dir / split_name / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            ann_count = 0
            class_dist: dict[str, int] = {c: 0 for c in self.class_names}

            for img_id in id_set:
                img_record = id_to_image[img_id]
                filename = img_record["file_name"]
                img_w = img_record["width"]
                img_h = img_record["height"]

                # Copy image
                src = self.images_dir / filename
                dst = img_dir / filename
                if not dst.exists():
                    shutil.copy2(src, dst)

                # Write YOLO label file
                anns = image_annotations[img_id]
                label_path = lbl_dir / (Path(filename).stem + ".txt")

                lines: list[str] = []
                for ann in anns:
                    cat_id = ann["category_id"]
                    if cat_id not in COCO_TO_MODEL_ID:
                        continue
                    model_id = COCO_TO_MODEL_ID[cat_id]
                    class_name = self.class_names[model_id]

                    x, y, w, h = ann["bbox"]  # COCO: [x_min, y_min, width, height]
                    if w <= 0 or h <= 0:
                        continue

                    # Convert to normalised centre format
                    cx = (x + w / 2) / img_w
                    cy = (y + h / 2) / img_h
                    nw = w / img_w
                    nh = h / img_h

                    # Clamp to [0, 1]
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    nw = max(0.0, min(1.0, nw))
                    nh = max(0.0, min(1.0, nh))

                    lines.append(f"{model_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                    ann_count += 1
                    class_dist[class_name] += 1

                label_path.write_text("\n".join(lines), encoding="utf-8")

            split_infos.append(
                SplitInfo(
                    name=split_name,
                    image_count=len(id_set),
                    annotation_count=ann_count,
                    class_distribution=class_dist,
                )
            )
            logger.info(
                "Split '%s': %d images, %d annotations",
                split_name, len(id_set), ann_count,
            )

        # 7. Write dataset.yaml
        dataset_yaml = self.prepared_dir / "dataset.yaml"
        yaml_content = (
            f"# VisionQC PCB Defect Detection Dataset\n"
            f"# Generated by src/visionqc/data/dataset_prep.py\n"
            f"# Seed: {self.seed}\n"
            f"\n"
            f"path: {self.prepared_dir.as_posix()}\n"
            f"train: train/images\n"
            f"val: val/images\n"
            f"test: test/images\n"
            f"\n"
            f"nc: {len(self.class_names)}\n"
            f"names: {self.class_names}\n"
        )
        dataset_yaml.write_text(yaml_content, encoding="utf-8")
        logger.info("dataset.yaml written to: %s", dataset_yaml)

        # 8. Count total annotations
        total_annotations = sum(s.annotation_count for s in split_infos)

        return DatasetStats(
            total_images=len(valid_image_ids),
            total_annotations=total_annotations,
            num_classes=len(self.class_names),
            splits=split_infos,
            seed=self.seed,
            split_ratios={
                "train": self.train_ratio,
                "val": self.val_ratio,
                "test": self.test_ratio,
            },
            prepared_dir=self.prepared_dir,
            dataset_yaml=dataset_yaml,
            warnings=warnings,
        )
