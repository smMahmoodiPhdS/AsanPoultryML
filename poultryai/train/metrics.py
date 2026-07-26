"""Training/validation metrics for multi-label disease detection.

Because onset windows are rare, we report threshold-free ranking metrics (PR-AUC /
average precision) as the primary signal, plus macro-F1 at a tuned threshold, plus
Expected Calibration Error (ECE) — calibration matters because operators act on the
probability, and an over-confident alarm erodes trust.
"""
from __future__ import annotations

import numpy as np
import torch


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve for one binary task (AP)."""
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    n_pos = labels.sum()
    if n_pos == 0:
        return float("nan")
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    # step integration over recall
    ap, prev_r = 0.0, 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_r); prev_r = r
    return float(ap)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Standard ECE (Guo et al., 2017) over the flattened multi-label predictions."""
    probs, labels = probs.ravel(), labels.ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs > lo) & (probs <= hi)
        if not m.any():
            continue
        conf = probs[m].mean()
        acc = labels[m].mean()
        ece += (m.mean()) * abs(acc - conf)
    return float(ece)


class MultiLabelMetrics:
    """Accumulate logits/labels across batches and compute epoch metrics."""

    def __init__(self, n_classes: int) -> None:
        self.n = n_classes
        self.reset()

    def reset(self) -> None:
        self._logits: list[np.ndarray] = []
        self._labels: list[np.ndarray] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        self._logits.append(logits.float().cpu().numpy())
        self._labels.append(labels.int().cpu().numpy())

    def compute(self, threshold: float = 0.5) -> dict[str, float]:
        if not self._logits:
            return {}
        logits = np.concatenate(self._logits)
        labels = np.concatenate(self._labels)
        probs = 1.0 / (1.0 + np.exp(-logits))
        aps = [average_precision(probs[:, c], labels[:, c]) for c in range(self.n)]
        macro_ap = float(np.nanmean(aps))
        preds = (probs >= threshold).astype(int)
        tp = (preds & labels).sum(0)
        fp = (preds & (1 - labels)).sum(0)
        fn = ((1 - preds) & labels).sum(0)
        f1 = (2 * tp / np.maximum(2 * tp + fp + fn, 1)).mean()
        return {"macro_ap": macro_ap, "macro_f1": float(f1),
                "ece": expected_calibration_error(probs, labels)}
