# -*- coding: utf-8 -*-
"""
Self-referenced depth-map consistency metrics, used by Extensions I and III.

Every metric here compares a "test" depth map back to a "reference" depth map
(the clean-image prediction for Ext. I, the ViT-L prediction for Ext. III) —
none of them requires external ground truth.
"""
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compare_depth_maps(ref_depth: np.ndarray, test_depth: np.ndarray) -> dict:
    """Compute MAE, RMSE, mean relative deviation, SSIM, and Agree@5% between
    a reference depth map and a test depth map.

    Returns:
        dict with keys MAE_m, RMSE_m, Rel_pct, SSIM, Agree_at_5pct.
    """
    if test_depth.shape != ref_depth.shape:
        test_depth = cv2.resize(test_depth, (ref_depth.shape[1], ref_depth.shape[0]))

    diff = test_depth - ref_depth
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    rel_pct = float(np.mean(np.abs(diff) / np.clip(ref_depth, 1e-3, None)) * 100)
    data_range = ref_depth.max() - ref_depth.min()
    ssim_val = float(ssim(ref_depth, test_depth, data_range=data_range))
    agree_5pct = float(np.mean(np.abs(diff) / np.clip(ref_depth, 1e-3, None) < 0.05) * 100)

    return dict(
        MAE_m=round(mae, 4),
        RMSE_m=round(rmse, 4),
        Rel_pct=round(rel_pct, 2),
        SSIM=round(ssim_val, 4),
        Agree_at_5pct=round(agree_5pct, 2),
    )
