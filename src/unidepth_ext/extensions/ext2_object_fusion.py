# -*- coding: utf-8 -*-
"""
Extension II — open-vocabulary object-level depth fusion, runnable outside Colab.

Usage:
    python -m unidepth_ext.extensions.ext2_object_fusion \
        --image path/to/photo.jpg --backbone vitl14 --out results/

Writes ext2_object_depths.csv and ext2_spatial_consistency.csv to --out.
Requires: pip install transformers (Grounding DINO + BLIP).
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

from unidepth_ext.detection import detect_objects, load_detection_models
from unidepth_ext.model import free_model, load_unidepth, run_inference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, help="Path to an RGB test photo.")
    p.add_argument("--backbone", default="vitl14", choices=["vits14", "vitb14", "vitl14"])
    p.add_argument("--out", default="results", help="Output directory for the CSVs.")
    p.add_argument("--score-thresh", type=float, default=0.30)
    p.add_argument("--text-thresh", type=float, default=0.25)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    depth_model = load_unidepth(args.backbone)
    gdino_processor, gdino_model, blip_processor, blip_model = load_detection_models()

    rgb = np.array(Image.open(args.image).convert("RGB"))
    depth, _ = run_inference(rgb, depth_model)
    dets = detect_objects(
        rgb, gdino_processor, gdino_model, blip_processor, blip_model,
        score_thresh=args.score_thresh, text_thresh=args.text_thresh,
    )

    rows = []
    for d in dets:
        x1, y1, x2, y2 = d["box"]
        patch = depth[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        rows.append(dict(
            image=os.path.basename(args.image),
            label=d["label"],
            score=round(d["score"], 3),
            median_depth_m=round(float(np.median(patch)), 3),
            box_bottom_y=y2,
            image_height=rgb.shape[0],
        ))

    df = pd.DataFrame(rows)
    depths_csv = os.path.join(args.out, "ext2_object_depths.csv")
    df.to_csv(depths_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved {depths_csv}")

    if len(df) >= 3:
        rho, pval = spearmanr(df["box_bottom_y"], df["median_depth_m"])
        cdf = pd.DataFrame([dict(
            image=os.path.basename(args.image), n_objects=len(df),
            spearman_rho=round(float(rho), 3), p_value=round(float(pval), 4),
        )])
        cons_csv = os.path.join(args.out, "ext2_spatial_consistency.csv")
        cdf.to_csv(cons_csv, index=False)
        print(cdf.to_string(index=False))
        print(f"Saved {cons_csv}")
    else:
        print("Fewer than 3 detections — skipping spatial-consistency correlation.")

    free_model(depth_model)


if __name__ == "__main__":
    main()
