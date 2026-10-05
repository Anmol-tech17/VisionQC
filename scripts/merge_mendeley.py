import sys
import shutil
import logging
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

MENDELEY_ROOT = _PROJECT_ROOT / "data" / "raw" / "mendeley" / "dataset" / "MIXED PCB DEFECT DETECTION"
PREPARED_TRAIN_DIR = _PROJECT_ROOT / "data" / "prepared" / "train"

def main():
    if not MENDELEY_ROOT.exists():
        logger.error("Mendeley dataset not found at %s", MENDELEY_ROOT)
        sys.exit(1)
        
    if not PREPARED_TRAIN_DIR.exists():
        logger.error("Prepared train dir not found. Run scripts/prepare_dataset.py first.")
        sys.exit(1)
        
    img_dst = PREPARED_TRAIN_DIR / "images"
    lbl_dst = PREPARED_TRAIN_DIR / "labels"
    
    total_images_copied = 0
    total_labels_copied = 0
    total_annotations_mapped = 0
    
    # We copy from train, valid, test of Mendeley all into our TRAIN split.
    for split in ["train", "valid", "test"]:
        split_dir = MENDELEY_ROOT / split
        if not split_dir.exists():
            continue
            
        logger.info("Processing Mendeley split: %s", split)
        
        for img_path in (split_dir / "images").glob("*.jpg"):
            # Copy image
            shutil.copy2(img_path, img_dst / img_path.name)
            total_images_copied += 1
            
            # Process corresponding label
            lbl_path = split_dir / "labels" / (img_path.stem + ".txt")
            if lbl_path.exists():
                lines = lbl_path.read_text(encoding="utf-8").strip().split("\n")
                new_lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    class_id = int(parts[0])
                    
                    # Map Mendeley class 0 (missing hole) to 6 (missing_hole in our mapping)
                    if class_id == 0:
                        class_id = 6
                        
                    new_lines.append(f"{class_id} " + " ".join(parts[1:]))
                    total_annotations_mapped += 1
                    
                (lbl_dst / lbl_path.name).write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                total_labels_copied += 1

    logger.info("Successfully merged Mendeley dataset into training pipeline.")
    logger.info("  Images copied: %d", total_images_copied)
    logger.info("  Labels copied: %d", total_labels_copied)
    logger.info("  Total annotations processed: %d", total_annotations_mapped)
    
if __name__ == "__main__":
    main()
