# -*- coding: utf-8 -*-
"""
Model loading and inference helpers for UniDepthV2.

Requires the official UniDepth repo to be installed first:
    git clone https://github.com/lpiccinelli-eth/UniDepth.git
    cd UniDepth && pip install -e .

See notebooks/UniDepth_Extensions_Colab.py, STEP 1, for the exact Colab setup sequence.
"""
import gc

import numpy as np
import torch

try:
    from unidepth.models import UniDepthV2
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Could not import unidepth. Install the official UniDepth repo first — "
        "see 'Installation' in the top-level README.md."
    ) from e

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The only three official UniDepthV2 checkpoints published on Hugging Face.
V2_BACKBONES = ["vits14", "vitb14", "vitl14"]


def load_unidepth(backbone: str):
    """Load a UniDepthV2 checkpoint by short name.

    Args:
        backbone: one of 'vits14', 'vitb14', 'vitl14'.
    """
    if backbone not in V2_BACKBONES:
        raise ValueError(f"backbone must be one of {V2_BACKBONES}, got {backbone!r}")
    m = UniDepthV2.from_pretrained(f"lpiccinelli/unidepth-v2-{backbone}")
    return m.to(device).eval()


def free_model(m):
    """Release a model's GPU memory (safe to call on CPU too)."""
    del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_inference(rgb_uint8: np.ndarray, model):
    """Run UniDepth on a single RGB image.

    Args:
        rgb_uint8: HxWx3 uint8 numpy array.
        model: a loaded UniDepthV2 model (see load_unidepth).

    Returns:
        (depth_map_metres, intrinsics_3x3) as numpy arrays.
    """
    rgb_t = torch.from_numpy(rgb_uint8).permute(2, 0, 1).to(device)
    with torch.no_grad():
        preds = model.infer(rgb_t)
    depth = preds["depth"].squeeze().cpu().numpy()
    intrinsics = preds["intrinsics"].squeeze().cpu().numpy()
    return depth, intrinsics
