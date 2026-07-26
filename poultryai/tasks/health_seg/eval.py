"""Health-seg metrics (numpy-only): per-class IoU/Dice, mIoU, pixel-acc + image-level health.

Same metric philosophy as V2 (``docs/metrics.md``) but over the health classes
``{background, healthy, sick}``. The derived **image-level** health call (dominant
non-background class) makes V4 comparable to a Healthy/Sick classifier and is the signal V3's
``distress_index`` and the fusion model consume.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import SEG_CLASSES, N_SEG_CLASSES


def update_confusion(conf: np.ndarray, pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    k = N_SEG_CLASSES
    idx = true.reshape(-1).astype(np.int64) * k + pred.reshape(-1).astype(np.int64)
    conf += np.bincount(idx, minlength=k * k).reshape(k, k)
    return conf


def seg_metrics(conf: np.ndarray) -> dict[str, Any]:
    conf = conf.astype(np.float64)
    tp = np.diag(conf)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1e-9)
    dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    present = conf.sum(1) > 0
    return {
        "per_class": {SEG_CLASSES[i]: {"iou": float(iou[i]), "dice": float(dice[i]),
                                       "pixels": int(conf[i].sum())}
                      for i in range(N_SEG_CLASSES)},
        "mIoU": float(iou[present].mean()) if present.any() else 0.0,
        "mIoU_foreground": float(iou[1:][present[1:]].mean()) if present[1:].any() else 0.0,
        "pixel_accuracy": float(tp.sum() / max(conf.sum(), 1e-9)),
    }


def dominant_health(pred_mask: np.ndarray) -> int:
    """Image-level class = most frequent non-background pixel (0 if none / background)."""
    counts = np.bincount(pred_mask.reshape(-1), minlength=N_SEG_CLASSES)
    counts[0] = 0
    return int(counts.argmax()) if counts.sum() > 0 else 0


def image_health_report(pred_labels: np.ndarray, true_labels: np.ndarray) -> dict[str, Any]:
    """Confusion over {background, healthy, sick} at image level + accuracy/recalls."""
    k = N_SEG_CLASSES
    conf = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(true_labels, pred_labels):
        conf[int(t), int(p)] += 1
    acc = float(np.trace(conf) / max(conf.sum(), 1))
    recalls = {SEG_CLASSES[i]: (float(conf[i, i] / conf[i].sum()) if conf[i].sum() else None)
               for i in range(k)}
    return {"accuracy": acc, "recall": recalls, "confusion_matrix": conf.tolist(),
            "classes": list(SEG_CLASSES)}


def format_seg_metrics(m: dict[str, Any]) -> str:
    lines = [f"mIoU={m['mIoU']:.4f}  mIoU_fg={m['mIoU_foreground']:.4f}  "
             f"pixel_acc={m['pixel_accuracy']:.4f}", "",
             f"{'class':12s} {'IoU':>7s} {'Dice':>7s} {'pixels':>12s}"]
    for c, v in m["per_class"].items():
        lines.append(f"{c:12s} {v['iou']:7.3f} {v['dice']:7.3f} {v['pixels']:12d}")
    return "\n".join(lines)
