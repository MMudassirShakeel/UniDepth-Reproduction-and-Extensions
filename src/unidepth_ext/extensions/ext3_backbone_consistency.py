# -*- coding: utf-8 -*-
"""
Extension III — cross-backbone consistency (ViT-S / ViT-B / ViT-L), runnable outside Colab.

Usage:
    python -m unidepth_ext.extensions.ext3_backbone_consistency \
        --image path/to/photo.jpg --out results/

Runs all three official UniDepthV2 backbones on the same image and reports
MAE/RMSE/SSIM/Agree@5% + a global median scale ratio, each vs. ViT-L (the
paper's largest, most accurate checkpoint) as reference.
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

from unidepth_ext.metrics import compare_depth_maps
from unidepth_ext.model import V2_BACKBONES, free_model, load_unidepth, run_inference

REFERENCE_BACKBONE = "vitl14"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, help="Path to an RGB test photo.")
    p.add_argument("--out", default="results", help="Output directory for the CSV.")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rgb = np.array(Image.open(args.image).convert("RGB"))

    depths = {}
    for bb in V2_BACKBONES:
        m = load_unidepth(bb)
        depth, _ = run_inference(rgb, m)
        depths[bb] = depth
        print(f"{bb:<8s} depth range {depth.min():.2f}-{depth.max():.2f} m")
        free_model(m)

    d_ref = depths[REFERENCE_BACKBONE]
    rows = []
    for bb in V2_BACKBONES:
        if bb == REFERENCE_BACKBONE:
            continue
        d_test = depths[bb]
        metrics = compare_depth_maps(d_ref, d_test)
        scale_ratio = float(np.median(d_test) / np.median(d_ref))
        rows.append(dict(
            image=os.path.basename(args.image), backbone=bb, reference=REFERENCE_BACKBONE,
            **metrics, median_scale_ratio_vs_ref=round(scale_ratio, 3),
        ))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out, "ext3_backbone_consistency.csv")
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
