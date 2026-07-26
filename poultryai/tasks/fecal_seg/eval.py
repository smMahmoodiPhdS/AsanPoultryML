"""Segmentation metrics for V2 (numpy-only; unit-testable without torch).

Reports per-class IoU and Dice, mean IoU (mIoU), and pixel accuracy — the standard semantic
segmentation metrics — plus a **derived image-level disease report** (predict the image's
disease as its dominant non-background class) so V2 is directly comparable to V1's macro-F1
and confusion matrix and to the thesis' metric philosophy (``docs/metrics.md``).

Usage: accumulate a confusion matrix over the pixel classes with :func:`update_confusion`,
then call :func:`seg_metrics`.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import SEG_CLASSES, N_SEG_CLASSES


def update_confusion(conf: np.ndarray, pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Accumulate an (K, K) pixel confusion matrix (rows=true, cols=pred)."""
    k = N_SEG_CLASSES
    idx = true.reshape(-1).astype(np.int64) * k + pred.reshape(-1).astype(np.int64)
    conf += np.bincount(idx, minlength=k * k).reshape(k, k)
    return conf


def seg_metrics(conf: np.ndarray) -> dict[str, Any]:
    """Per-class IoU/Dice, mIoU, pixel accuracy from a pixel confusion matrix."""
    conf = conf.astype(np.float64)
    tp = np.diag(conf)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1e-9)
    dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    present = (conf.sum(1) > 0)
    out: dict[str, Any] = {
        "per_class": {SEG_CLASSES[i]: {"iou": float(iou[i]), "dice": float(dice[i]),
                                       "pixels": int(conf[i].sum())}
                      for i in range(N_SEG_CLASSES)},
        "mIoU": float(iou[present].mean()) if present.any() else 0.0,
        "mIoU_foreground": float(iou[1:][present[1:]].mean()) if present[1:].any() else 0.0,
        "pixel_accuracy": float(tp.sum() / max(conf.sum(), 1e-9)),
    }
    return out


def dominant_class(pred_mask: np.ndarray) -> int:
    """Image-level disease = most frequent non-background pixel class (0 if none)."""
    counts = np.bincount(pred_mask.reshape(-1), minlength=N_SEG_CLASSES)
    counts[0] = 0
    return int(counts.argmax()) if counts.sum() > 0 else 0


def format_seg_metrics(m: dict[str, Any]) -> str:
    lines = [f"mIoU={m['mIoU']:.4f}  mIoU_fg={m['mIoU_foreground']:.4f}  "
             f"pixel_acc={m['pixel_accuracy']:.4f}", "",
             f"{'class':14s} {'IoU':>7s} {'Dice':>7s} {'pixels':>12s}"]
    for c, v in m["per_class"].items():
        lines.append(f"{c:14s} {v['iou']:7.3f} {v['dice']:7.3f} {v['pixels']:12d}")
    return "\n".join(lines)
