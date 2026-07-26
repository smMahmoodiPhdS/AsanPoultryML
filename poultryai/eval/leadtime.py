"""Lead-time evaluation — the clinically meaningful metric for this thesis.

Window accuracy alone does not answer the research question ("how much *earlier* than a
human?"). We evaluate, per flock and per disease, the **detection lead time**: the time
between the first confident alarm and the veterinary-confirmed onset, subject to a
false-alarm budget.

Protocol (per disease d, per flock f)
-------------------------------------
1. Order the flock's windows by time. Each window has a score s_t in [0,1] and an
   ``onset_offset`` o_t = seconds from window end to confirmed onset (+inf if healthy).
2. An **alarm** fires at the first time the score stays >= ``threshold`` for
   ``persistence`` consecutive windows (debounced, to suppress single-window spikes).
3. If the flock is diseased and the alarm fires at or before onset, lead_time = seconds
   between alarm time and onset (>= 0). An alarm strictly after onset counts as a
   *late* detection (lead_time <= 0) — detected but not early.
4. If the flock is healthy and any alarm fires, it is a **false positive flock**.
5. The operating threshold is chosen on the validation set to hold the flock-level
   false-alarm rate at/under a target (e.g. <= 5%), then frozen for test.

Reported: median lead time (h), % detected-before-onset, flock-level FPR, and lead time
vs. a human-supervisor baseline (supplied as onset-minus-inspection timestamps).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FlockSeries:
    times: np.ndarray          # (T,) epoch seconds, ascending
    scores: np.ndarray         # (T,) probability for one disease
    onset_offset: np.ndarray   # (T,) seconds from window end to onset (+inf if healthy)

    @property
    def is_diseased(self) -> bool:
        return np.isfinite(self.onset_offset).any()

    def onset_time(self) -> float:
        # window_end + offset is constant across windows for a real onset; recover it.
        finite = np.isfinite(self.onset_offset)
        if not finite.any():
            return float("inf")
        idx = np.argmax(finite)
        return float(self.times[idx] + self.onset_offset[idx])


def first_alarm_time(series: FlockSeries, threshold: float, persistence: int = 3) -> float:
    """Time of the first debounced alarm, or +inf if none."""
    above = series.scores >= threshold
    run = 0
    for t, a in zip(series.times, above):
        run = run + 1 if a else 0
        if run >= persistence:
            return float(t)
    return float("inf")


@dataclass
class LeadTimeResult:
    median_lead_s: float
    detected_before_onset: float   # fraction of diseased flocks alarmed at/before onset
    flock_false_positive_rate: float
    n_diseased: int
    n_healthy: int


def evaluate_leadtime(flocks: list[FlockSeries], threshold: float,
                      persistence: int = 3) -> LeadTimeResult:
    leads: list[float] = []
    detected = 0
    diseased = [f for f in flocks if f.is_diseased]
    healthy = [f for f in flocks if not f.is_diseased]

    for f in diseased:
        alarm = first_alarm_time(f, threshold, persistence)
        onset = f.onset_time()
        if np.isfinite(alarm):
            lead = onset - alarm            # >0 => early, <=0 => late
            leads.append(lead)
            if alarm <= onset:
                detected += 1

    fp = sum(1 for f in healthy if np.isfinite(first_alarm_time(f, threshold, persistence)))
    return LeadTimeResult(
        median_lead_s=float(np.median(leads)) if leads else float("nan"),
        detected_before_onset=detected / len(diseased) if diseased else float("nan"),
        flock_false_positive_rate=fp / len(healthy) if healthy else float("nan"),
        n_diseased=len(diseased), n_healthy=len(healthy),
    )


def tune_threshold_for_fpr(flocks: list[FlockSeries], target_fpr: float = 0.05,
                           persistence: int = 3, grid: int = 101) -> float:
    """Pick the lowest threshold whose flock-level FPR <= target (maximising lead time).
    Lower threshold => earlier alarms => more lead time but more false positives, so we
    take the most sensitive threshold that still respects the false-alarm budget."""
    healthy = [f for f in flocks if not f.is_diseased]
    if not healthy:
        return 0.5
    for thr in np.linspace(0.0, 1.0, grid):
        fp = sum(1 for f in healthy
                 if np.isfinite(first_alarm_time(f, float(thr), persistence)))
        if fp / len(healthy) <= target_fpr:
            return float(thr)
    return 1.0
