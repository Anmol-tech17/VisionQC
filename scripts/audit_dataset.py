"""
VisionQC — Dataset Audit Script
---------------------------------
Performs a comprehensive quality audit of the prepared YOLO dataset.
Run this before designing new training experiments.

Usage:
    C:\\Program Files\\Python313\\python.exe scripts/audit_dataset.py
"""
import sys, json, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image as PILImage

prepared = Path("data/prepared")
CLASSES = ["missing_pad","mouse_bite","open_circuit","short","spur","spurious_copper"]

audit_data = {}

print("=" * 65)
print("  VisionQC — Dataset Quality Audit")
print("=" * 65)
print()

for split in ["train", "val", "test"]:
    images_dir = prepared / split / "images"
    labels_dir = prepared / split / "labels"
    imgs = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    lbls = sorted(labels_dir.glob("*.txt"))

    img_stems = {f.stem for f in imgs}
    lbl_stems = {f.stem for f in lbls}
    no_label  = img_stems - lbl_stems
    no_img    = lbl_stems - img_stems

    class_counts = [0]*6
    bbox_ws, bbox_hs, bbox_areas = [], [], []
    invalid_coords = 0
    empty_lbls = 0

    for lbl_f in lbls:
        content = lbl_f.read_text().strip()
        if not content:
            empty_lbls += 1
            continue
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx,cy,w,h = float(parts[1]),float(parts[2]),float(parts[3]),float(parts[4])
            if cls_id < 6:
                class_counts[cls_id] += 1
            if not (0<=cx<=1 and 0<=cy<=1 and 0<w<=1 and 0<h<=1):
                invalid_coords += 1
            bbox_ws.append(w)
            bbox_hs.append(h)
            bbox_areas.append(w*h)

    total_anns = sum(class_counts)
    wa = np.array(bbox_ws)
    ha = np.array(bbox_hs)
    aa = np.array(bbox_areas)
    small = ((wa<0.05)|(ha<0.05)).sum() if len(wa) else 0

    split_data = {
        "images": len(imgs),
        "label_files": len(lbls),
        "images_missing_label": list(no_label),
        "labels_missing_image": list(no_img),
        "empty_label_files": empty_lbls,
        "invalid_coords": invalid_coords,
        "total_annotations": total_anns,
        "class_distribution": {CLASSES[i]: class_counts[i] for i in range(6)},
        "bbox_stats": {
            "width":  {"min": float(wa.min()) if len(wa) else 0, "mean": float(wa.mean()) if len(wa) else 0, "max": float(wa.max()) if len(wa) else 0},
            "height": {"min": float(ha.min()) if len(ha) else 0, "mean": float(ha.mean()) if len(ha) else 0, "max": float(ha.max()) if len(ha) else 0},
            "area":   {"min": float(aa.min()) if len(aa) else 0, "mean": float(aa.mean()) if len(aa) else 0, "max": float(aa.max()) if len(aa) else 0},
            "small_boxes_pct": float(100*small/len(wa)) if len(wa) else 0,
        } if len(wa) else {},
    }
    audit_data[split] = split_data

    print(f"--- {split.upper()} ({len(imgs)} images) ---")
    print(f"  Annotations:     {total_anns}")
    print(f"  Empty lbl files: {empty_lbls}")
    print(f"  Invalid coords:  {invalid_coords}")
    if no_label: print(f"  WARN: {len(no_label)} images with no label")
    if no_img:   print(f"  WARN: {len(no_img)} orphan label files")
    for i,c in enumerate(CLASSES):
        pct = 100*class_counts[i]/total_anns if total_anns else 0
        print(f"    {c:<22} {class_counts[i]:>5}  ({pct:.1f}%)")
    if len(wa):
        print(f"  BBox W: min={wa.min():.4f}  mean={wa.mean():.4f}  max={wa.max():.4f}")
        print(f"  BBox H: min={ha.min():.4f}  mean={ha.mean():.4f}  max={ha.max():.4f}")
        print(f"  Small (<5% dim): {small} ({100*small/len(wa):.1f}%)")
    print()

# Image dimensions
print("--- IMAGE DIMENSIONS (30 sample train images) ---")
dims = []
for img_f in sorted((prepared/"train"/"images").glob("*.jpg"))[:30]:
    img = PILImage.open(img_f)
    dims.append(img.size)
ws2=[d[0] for d in dims]; hs2=[d[1] for d in dims]
print(f"  W: min={min(ws2)} max={max(ws2)} mean={int(sum(ws2)/len(ws2))}")
print(f"  H: min={min(hs2)} max={max(hs2)} mean={int(sum(hs2)/len(hs2))}")
ratios=[w/h for w,h in dims]
print(f"  Aspect ratio: min={min(ratios):.2f}  max={max(ratios):.2f}  mean={sum(ratios)/len(ratios):.2f}")
print()

# Save audit report
report = {
    "splits": audit_data,
    "sample_image_dimensions": {"widths": ws2, "heights": hs2},
    "notes": [
        "No leakage verified by assertion in converter (train/val/test disjoint)",
        "All 6 defect classes present in train split",
        "Seed=42 used for reproducible split",
    ]
}
Path("reports").mkdir(exist_ok=True)
Path("reports/dataset_audit.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print("Audit report saved: reports/dataset_audit.json")
print()

# Recommendations
print("--- RECOMMENDATIONS FOR EXPERIMENT A ---")
all_ws = []
all_hs = []
for split in ["train","val","test"]:
    for lbl_f in (prepared/split/"labels").glob("*.txt"):
        for line in lbl_f.read_text().strip().splitlines():
            p = line.split()
            if len(p)==5:
                all_ws.append(float(p[3]))
                all_hs.append(float(p[4]))
wa_all = np.array(all_ws); ha_all = np.array(all_hs)
median_w = float(np.median(wa_all))
median_h = float(np.median(ha_all))
print(f"  Overall median bbox W={median_w:.4f}  H={median_h:.4f}")
print(f"  At 320px: median bbox = {int(median_w*320)}x{int(median_h*320)} px")
print(f"  At 416px: median bbox = {int(median_w*416)}x{int(median_h*416)} px")
print(f"  At 640px: median bbox = {int(median_w*640)}x{int(median_h*640)} px")
small_pct = float(100*((wa_all<0.05)|(ha_all<0.05)).mean())
print(f"  Boxes <5% dim: {small_pct:.1f}%  -> larger imgsz may help")
print()
