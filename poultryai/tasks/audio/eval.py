"""Audio metrics: window-level and recording-level reports (numpy/sklearn).

Following ``docs/metrics.md``: imbalance-robust ranking metrics + calibration, not accuracy.
The **recording-level** report is the honest unit — a farmer cares about the bird/recording,
not the 4 s window — so we aggregate window probabilities per recording (mean) before scoring.
Reuses the calibration math from the fecal task.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from . import CLASSES
from poultryai.tasks.fecal.eval import compute_ece


def _report(probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    probs = np.asarray(probs, float); labels = np.asarray(labels)
    pred = probs.argmax(1)
    out: dict[str, Any] = {"n": int(len(labels)), "classes": list(CLASSES),
                           "accuracy": float((pred == labels).mean()),
                           "ece": compute_ece(probs, labels)}
    try:
        from sklearn.metrics import (confusion_matrix, f1_score,
                                     precision_recall_fscore_support, roc_auc_score)
        p, r, f1, sup = precision_recall_fscore_support(
            labels, pred, labels=range(len(CLASSES)), zero_division=0)
        out["per_class"] = {CLASSES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                         "f1": float(f1[i]), "support": int(sup[i])}
                            for i in range(len(CLASSES))}
        out["macro_f1"] = float(f1_score(labels, pred, average="macro", zero_division=0))
        # binary health view (healthy vs not-healthy), the clinically relevant collapse
        health_true = (labels == 0).astype(int)
        health_score = probs[:, 0]
        try:
            out["macro_auroc"] = float(roc_auc_score(
                labels, probs, multi_class="ovr", average="macro",
                labels=list(range(len(CLASSES)))))
            out["healthy_auroc"] = float(roc_auc_score(health_true, health_score))
        except Exception:
            out["macro_auroc"] = out["healthy_auroc"] = None
        out["confusion_matrix"] = confusion_matrix(
            labels, pred, labels=range(len(CLASSES))).tolist()
    except Exception:
        out["note"] = "sklearn unavailable; install scikit-learn for full metrics"
    return out


def window_report(probs: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    return _report(probs, labels)


def recording_report(probs: np.ndarray, labels: np.ndarray,
                     rec_ids: list[str]) -> dict[str, Any]:
    """Aggregate window probabilities by recording (mean) then score — the honest unit."""
    agg_p: dict[str, list[np.ndarray]] = defaultdict(list)
    agg_y: dict[str, int] = {}
    for p, y, rid in zip(np.asarray(probs, float), np.asarray(labels), rec_ids):
        agg_p[rid].append(p); agg_y[rid] = int(y)
    ids = list(agg_p)
    P = np.stack([np.mean(agg_p[i], axis=0) for i in ids])
    Y = np.array([agg_y[i] for i in ids])
    rep = _report(P, Y)
    rep["n_recordings"] = len(ids)
    return rep


def format_report(rep: dict[str, Any]) -> str:
    lines = [f"n={rep['n']}  acc={rep['accuracy']:.4f}  "
             f"macro_f1={rep.get('macro_f1', float('nan')):.4f}  "
             f"macro_auroc={rep.get('macro_auroc')}  healthy_auroc={rep.get('healthy_auroc')}  "
             f"ece={rep['ece']:.4f}"]
    if "per_class" in rep:
        lines += ["", f"{'class':10s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'n':>6s}"]
        for c, m in rep["per_class"].items():
            lines.append(f"{c:10s} {m['precision']:6.3f} {m['recall']:6.3f} "
                         f"{m['f1']:6.3f} {m['support']:6d}")
    if "confusion_matrix" in rep:
        lines += ["", "confusion (rows=true, cols=pred): " + str(rep["confusion_matrix"])]
    return "\n".join(lines)
