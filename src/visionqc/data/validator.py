"""
VisionQC — COCO Dataset Validator
-----------------------------------
Validates the PCB defect dataset before use.

Usage
-----
    from src.visionqc.data.validator import DatasetValidator

    validator = DatasetValidator()
    result = validator.validate()
    if result.passed:
        print(result.summary_text())
    else:
        for err in result.errors:
            print("ERROR:", err)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from configs.settings import ANNOTATION_FILE, DEFECT_CLASSES, IMAGES_DIR


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    passed: bool
    image_count: int = 0
    annotation_count: int = 0
    defect_class_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary_text(self) -> str:
        if self.passed:
            lines = [
                "DATASET VALIDATION PASSED",
                "",
                f"  Images:          {self.image_count}",
                f"  Annotations:     {self.annotation_count}",
                f"  Defect classes:  {self.defect_class_count}",
            ]
            if self.warnings:
                lines.append("")
                for w in self.warnings:
                    lines.append(f"  WARNING: {w}")
        else:
            lines = ["DATASET VALIDATION FAILED", ""]
            for e in self.errors:
                lines.append(f"  ERROR: {e}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "image_count": self.image_count,
            "annotation_count": self.annotation_count,
            "defect_class_count": self.defect_class_count,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class DatasetValidator:
    """
    Validates the COCO-format PCB defect dataset.

    Parameters
    ----------
    annotation_file : Path, optional
        Path to _annotations.coco.json.  Defaults to the value in settings.
    images_dir : Path, optional
        Path to the images directory.  Defaults to the value in settings.
    """

    def __init__(
        self,
        annotation_file: Optional[Path] = None,
        images_dir: Optional[Path] = None,
    ) -> None:
        self.annotation_file = annotation_file or ANNOTATION_FILE
        self.images_dir = images_dir or IMAGES_DIR

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def validate(self) -> ValidationResult:
        """Run all checks and return a ValidationResult."""
        errors: list[str] = []
        warnings: list[str] = []

        # 1. Annotation file exists
        if not self.annotation_file.exists():
            errors.append(
                f"Annotation file not found: {self.annotation_file}"
            )
            return ValidationResult(passed=False, errors=errors)

        # 2. Images directory exists
        if not self.images_dir.exists():
            errors.append(
                f"Images directory not found: {self.images_dir}"
            )
            return ValidationResult(passed=False, errors=errors)

        # 3. Parse the COCO JSON
        try:
            with open(self.annotation_file, encoding="utf-8") as f:
                coco = json.load(f)
        except json.JSONDecodeError as exc:
            errors.append(f"Failed to parse annotation JSON: {exc}")
            return ValidationResult(passed=False, errors=errors)

        # 4. Required top-level keys
        for key in ("images", "annotations", "categories"):
            if key not in coco:
                errors.append(f"Annotation JSON missing key: '{key}'")
        if errors:
            return ValidationResult(passed=False, errors=errors)

        images = coco["images"]
        annotations = coco["annotations"]
        categories = coco["categories"]

        # 5. Build valid image ID set
        valid_image_ids: set[int] = set()
        missing_images: list[str] = []
        for img in images:
            img_path = self.images_dir / img["file_name"]
            if img_path.exists():
                valid_image_ids.add(img["id"])
            else:
                missing_images.append(img["file_name"])

        if missing_images:
            sample = missing_images[:5]
            warnings.append(
                f"{len(missing_images)} image file(s) listed in annotations "
                f"not found on disk (showing first 5): {sample}"
            )

        # 6. Validate category IDs — category 0 is root label, skip it
        valid_category_ids = set(DEFECT_CLASSES.keys())  # {1..6}
        unknown_category_ids: set[int] = set()
        for cat in categories:
            cat_id = cat["id"]
            if cat_id != 0 and cat_id not in valid_category_ids:
                unknown_category_ids.add(cat_id)
        if unknown_category_ids:
            warnings.append(
                f"Unknown category IDs (not in 1–6): {unknown_category_ids}"
            )

        # 7. Validate bounding boxes
        bad_bbox_count = 0
        orphan_annotation_count = 0
        for ann in annotations:
            # Check image ID is valid
            if ann["image_id"] not in valid_image_ids:
                orphan_annotation_count += 1
                continue

            # Skip root category (0) annotations
            if ann.get("category_id", 0) == 0:
                continue

            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                bad_bbox_count += 1
                continue
            _x, _y, w, h = bbox
            if w <= 0 or h <= 0:
                bad_bbox_count += 1

        if bad_bbox_count:
            errors.append(
                f"{bad_bbox_count} annotation(s) have invalid bounding boxes "
                "(width or height ≤ 0)."
            )
        if orphan_annotation_count:
            warnings.append(
                f"{orphan_annotation_count} annotation(s) reference image IDs "
                "not found in the images list."
            )

        # 8. Defect class count (categories excluding root)
        real_classes = [c for c in categories if c["id"] != 0]
        defect_class_count = len(real_classes)

        if defect_class_count == 0:
            errors.append("No defect classes found in the annotation file.")

        passed = len(errors) == 0

        return ValidationResult(
            passed=passed,
            image_count=len(images),
            annotation_count=len(annotations),
            defect_class_count=defect_class_count,
            errors=errors,
            warnings=warnings,
        )


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = DatasetValidator().validate()
    print(result.summary_text())
