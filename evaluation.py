"""Compare saved masks in run/ against ground truth polygons (pixel-wise IoU, original resolution).
Detected = IoU > 0.6. Reports detection rate and mean IoU of detected frames."""
import argparse
import json
import os

import cv2
import numpy as np

from dataset import list_images, label_path_for, polygons_to_mask, read_polygons

DEFAULT_DATA = "/Users/pittachiox/Downloads/data/1000/dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default=f"{DEFAULT_DATA}/images/test")
    ap.add_argument("--pred-dir", default="run")
    ap.add_argument("--thr", type=float, default=0.6)
    ap.add_argument("--out", default="evaluation_results.json")
    a = ap.parse_args()

    rows = []
    for f in list_images(a.img_dir):
        name = os.path.splitext(os.path.basename(f))[0]
        pred = cv2.imread(os.path.join(a.pred_dir, name + ".png"), cv2.IMREAD_GRAYSCALE)
        if pred is None:
            continue
        pred = pred > 127
        gt = polygons_to_mask(read_polygons(label_path_for(f)), *pred.shape).astype(bool)
        union = (pred | gt).sum()
        iou = float((pred & gt).sum() / union) if union else 1.0
        rows.append({"image": name, "iou": iou, "detected": iou > a.thr})
    ious = np.array([r["iou"] for r in rows])
    det = np.array([r["detected"] for r in rows])
    summary = {
        "num_images": len(rows),
        "iou_threshold": a.thr,
        "detected": int(det.sum()),
        "detection_rate": float(det.mean()),
        "mean_iou_detected": float(ious[det].mean()) if det.any() else 0.0,
        "mean_iou_all": float(ious.mean()),
    }
    print(json.dumps(summary, indent=2))
    json.dump({"summary": summary, "per_image": rows}, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
