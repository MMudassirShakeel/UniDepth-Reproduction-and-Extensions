# -*- coding: utf-8 -*-
"""
Extension I — robustness sweep, runnable outside Colab.

Usage:
    python -m unidepth_ext.extensions.ext1_robustness \
        --image path/to/photo.jpg --backbone vits14 --out results/

Writes ext1_robustness_results.csv (one row per degraded variant) to --out.
For the full qualitative-grid and trend-plot figures, see the Colab notebook
(notebooks/UniDepth_Extensions_Colab.py), which also handles multi-image batches.
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

from unidepth_ext.degradations import DEGRADATIONS
from unidepth_ext.metrics import compare_depth_maps
from unidepth_ext.model import free_model, load_unidepth, run_inference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True, help="Path to a clean RGB test photo.")
    p.add_argument("--backbone", default="vits14", choices=["vits14", "vitb14", "vitl14"])
    p.add_argument("--out", default="results", help="Output directory for the CSV.")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    model = load_unidepth(args.backbone)

    rgb = np.array(Image.open(args.image).convert("RGB"))
    depths = {}
    for name, fn in DEGRADATIONS.items():
        variant_rgb = fn(rgb.copy())
        depth, _ = run_inference(variant_rgb, model)
        depths[name] = depth
        print(f"{name:<16s} depth range {depth.min():.2f}-{depth.max():.2f} m")

    clean_depth = depths["clean"]
    rows = [
        dict(image=os.path.basename(args.image), variant=name, **compare_depth_maps(clean_depth, d))
        for name, d in depths.items() if name != "clean"
    ]
    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out, "ext1_robustness_results.csv")
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved {out_csv}")

    free_model(model)


if __name__ == "__main__":
    main()
