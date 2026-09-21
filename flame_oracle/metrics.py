from __future__ import annotations

import math

import numpy as np


def masked_image_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    mask = mask.astype(np.float32)
    if mask.ndim == 2:
        mask = mask[..., None]
    denom = max(float(mask.sum()) * pred.shape[-1], 1.0)
    difference = (pred - target) * mask
    mse = float(np.square(difference).sum() / denom)
    mae = float(np.abs(difference).sum() / denom)
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    return {"masked_mse": mse, "masked_mae": mae, "masked_psnr": psnr}


def alpha_metrics(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred_binary = pred >= threshold
    target_binary = target >= threshold
    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()
    return {
        "alpha_mae": float(np.abs(pred - target).mean()),
        "alpha_iou": float(intersection / max(union, 1)),
    }

