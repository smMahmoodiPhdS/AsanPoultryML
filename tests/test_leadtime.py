"""Unit tests for the lead-time evaluation logic (numpy-only, no torch needed)."""
import numpy as np

from poultryai.eval.leadtime import (
    FlockSeries, first_alarm_time, evaluate_leadtime, tune_threshold_for_fpr,
)

MIN = 60.0


def _series(scores, onset_idx=None, start=0.0, step=MIN):
    n = len(scores)
    times = start + step * np.arange(n)
    off = np.full(n, np.inf)
    if onset_idx is not None:
        onset_time = times[onset_idx]
        off = onset_time - times            # seconds from each window to onset
    return FlockSeries(times=times, scores=np.asarray(scores, float), onset_offset=off)


def test_debounce_requires_persistence():
    # single spike must NOT alarm at persistence=3
    s = _series([0.1, 0.9, 0.1, 0.1])
    assert not np.isfinite(first_alarm_time(s, 0.5, persistence=3))
    # three in a row alarms, at the 3rd sample (t=2*MIN)
    s2 = _series([0.1, 0.9, 0.9, 0.9])
    assert first_alarm_time(s2, 0.5, persistence=3) == 3 * MIN  # index 3 completes the run


def test_lead_time_is_positive_when_early():
    # onset at index 8; scores ramp so a debounced alarm fires around index 5
    scores = [0.1] * 4 + [0.9] * 6
    s = _series(scores, onset_idx=8)
    res = evaluate_leadtime([s], threshold=0.5, persistence=3)
    assert res.n_diseased == 1 and res.n_healthy == 0
    # alarm completes at index 6 (t=6*MIN); onset at index 8 (t=8*MIN) => +2 windows
    assert res.median_lead_s == 2 * MIN
    assert res.detected_before_onset == 1.0


def test_false_positive_flock_counted():
    healthy = _series([0.9, 0.9, 0.9, 0.9])          # alarms though healthy
    quiet_healthy = _series([0.0, 0.0, 0.0, 0.0])
    res = evaluate_leadtime([healthy, quiet_healthy], threshold=0.5, persistence=3)
    assert res.n_healthy == 2
    assert res.flock_false_positive_rate == 0.5


def test_threshold_tuning_respects_fpr_budget():
    # one noisy healthy flock; a high threshold should keep FPR under budget
    noisy = _series([0.6, 0.6, 0.6, 0.6])
    thr = tune_threshold_for_fpr([noisy], target_fpr=0.0, persistence=3)
    assert thr > 0.6   # must exceed the noise level to avoid the false alarm


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
