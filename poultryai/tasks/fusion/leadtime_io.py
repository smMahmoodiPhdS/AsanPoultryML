"""Turn windows + per-window scores into per-flock, per-disease ``FlockSeries`` (torch-free).

Bridges the model's per-window disease scores to ``eval.leadtime``: group windows by flock,
order by time, and emit one ``FlockSeries`` per (flock, disease) carrying the score track and
the ``onset_offset`` track. Also provides a **signature-proxy** score so the whole lead-time
protocol can be exercised without a trained model (a sanity oracle, not a result).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from poultryai.data.schema import DISEASES, N_DISEASES, Window
from poultryai.eval.leadtime import FlockSeries


def flock_key(w: Window) -> tuple[str, str]:
    return (w.meta.owner, w.meta.saloon)


def build_flock_series(windows: list[Window], scores: np.ndarray,
                       disease_idx: int) -> list[FlockSeries]:
    """One FlockSeries per flock for a given disease. ``scores`` is (n_windows, N_DISEASES)."""
    by_flock: dict[tuple, list[int]] = defaultdict(list)
    for i, w in enumerate(windows):
        by_flock[flock_key(w)].append(i)
    out: list[FlockSeries] = []
    for _, idxs in by_flock.items():
        idxs = sorted(idxs, key=lambda i: windows[i].t_end)
        times = np.array([windows[i].t_end for i in idxs], dtype=np.float64)
        sc = np.array([scores[i, disease_idx] for i in idxs], dtype=np.float64)
        oo = np.array([windows[i].onset_offset[disease_idx] for i in idxs], dtype=np.float64)
        out.append(FlockSeries(times=times, scores=sc, onset_offset=oo))
    return out


def signature_proxy_scores(windows: list[Window], sharpness: float = 6.0) -> np.ndarray:
    """A cheap oracle score from onset proximity — for validating the pipeline pre-training.

    Score rises as a window approaches onset (pre-onset horizon) and stays high after. Healthy
    flocks (onset_offset = +inf) score ~0. NOT a model — only to prove the lead-time protocol
    yields sensible, positive lead times end-to-end.
    """
    n = len(windows)
    scores = np.zeros((n, N_DISEASES), dtype=np.float64)
    horizon_s = 72.0 * 3600.0
    for i, w in enumerate(windows):
        for d in range(N_DISEASES):
            off = w.onset_offset[d]
            if np.isinf(off):
                continue
            # off>0 pre-onset (approaching), off<=0 post-onset (clinical, high)
            x = 1.0 - max(0.0, off) / horizon_s
            scores[i, d] = 1.0 / (1.0 + np.exp(-sharpness * (x - 0.6)))
    return scores
