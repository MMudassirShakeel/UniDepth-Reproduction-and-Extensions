# -*- coding: utf-8 -*-
"""
Image degradation functions used by Extension I (robustness sweep).

Three families at three severities each, matching Table III of the report:
low-light darkening, additive Gaussian noise, and horizontal motion blur.
"""
import cv2
import numpy as np


def darken(img: np.ndarray, gamma: float) -> np.ndarray:
    """Inverse-gamma darkening. Lower gamma = darker image."""
    return np.clip(255 * ((img / 255.0) ** (1.0 / gamma)), 0, 255).astype(np.uint8)


def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    """Additive Gaussian noise in normalized [0, 1] pixel units."""
    noisy = img / 255.0 + np.random.normal(0, sigma, img.shape)
    return np.clip(noisy * 255, 0, 255).astype(np.uint8)


def motion_blur(img: np.ndarray, k: int) -> np.ndarray:
    """Horizontal motion blur with a k x k averaging kernel."""
    kernel = np.zeros((k, k))
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


# The nine-condition sweep used throughout Extension I, plus the clean baseline.
DEGRADATIONS = {
    "clean": lambda im: im,
    "low_light_g0.7": lambda im: darken(im, 0.7),
    "low_light_g0.5": lambda im: darken(im, 0.5),
    "low_light_g0.3": lambda im: darken(im, 0.3),
    "noise_s0.02": lambda im: add_gaussian_noise(im, 0.02),
    "noise_s0.05": lambda im: add_gaussian_noise(im, 0.05),
    "noise_s0.10": lambda im: add_gaussian_noise(im, 0.10),
    "blur_k5": lambda im: motion_blur(im, 5),
    "blur_k9": lambda im: motion_blur(im, 9),
    "blur_k15": lambda im: motion_blur(im, 15),
}
