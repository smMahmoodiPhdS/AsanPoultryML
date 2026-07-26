"""Evaluation report for the fecal classifier.

Consistent with the thesis metric philosophy (``docs/metrics.md``): imbalance-robust
ranking metrics + calibration, never bare accuracy. Reports per-class precision/recall/F1,
macro-F1, macro AUROC, a confusion matrix, and **Expected Calibration Error** (operators
act on the probability). ``compute_ece`` is numpy-only so it is unit-testable without torch.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import CLASSES


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error (top-label, 15-bin) — matches metrics.py convention."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def classification_report(probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Full report dict. Uses sklearn when available; degrades gracefully otherwise."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels)
    pred = probs.argmax(axis=1)
    out: dict[str, Any] = {"n": int(len(labels)), "classes": list(CLASSES)}

    try:
        from sklearn.metrics import (confusion_matrix, f1_score,
                                     precision_recall_fscore_support, roc_auc_score)
        p, r, f1, sup = precision_recall_fscore_support(
            labels, pred, labels=range(len(CLASSES)), zero_division=0)
        out["per_class"] = {
            CLASSES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                         "f1": float(f1[i]), "support": int(sup[i])}
            for i in range(len(CLASSES))
        }
        out["macro_f1"] = float(f1_score(labels, pred, average="macro", zero_division=0))
        try:
            out["macro_auroc"] = float(roc_auc_score(
                labels, probs, multi_class="ovr", average="macro",
                labels=list(range(len(CLASSES)))))
        except Exception:
            out["macro_auroc"] = None
        out["confusion_matrix"] = confusion_matrix(
            labels, pred, labels=range(len(CLASSES))).tolist()
    except Exception:  # sklearn missing — minimal fallback
        acc = float((pred == labels).mean())
        out["accuracy"] = acc
        out["note"] = "sklearn unavailable; install scikit-learn for full metrics"

    out["ece"] = compute_ece(probs, labels)
    out["accuracy"] = float((pred == labels).mean())
    return out


def format_report(rep: dict[str, Any]) -> str:
    lines = [f"n={rep['n']}  accuracy={rep.get('accuracy', float('nan')):.4f}  "
             f"macro_f1={rep.get('macro_f1', float('nan')):.4f}  "
             f"macro_auroc={rep.get('macro_auroc')}  ece={rep['ece']:.4f}", ""]
    if "per_class" in rep:
        lines.append(f"{'class':14s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'n':>6s}")
        for c, m in rep["per_class"].items():
            lines.append(f"{c:14s} {m['precision']:6.3f} {m['recall']:6.3f} "
                         f"{m['f1']:6.3f} {m['support']:6d}")
    if "confusion_matrix" in rep:
        lines += ["", "confusion matrix (rows=true, cols=pred):",
                  "            " + " ".join(f"{c[:5]:>6s}" for c in rep["classes"])]
        for c, row in zip(rep["classes"], rep["confusion_matrix"]):
            lines.append(f"{c:12s}" + " ".join(f"{v:6d}" for v in row))
    return "\n".join(lines)
