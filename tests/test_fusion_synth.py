"""Torch-free tests for C1: synthetic window validity + the lead-time pipeline end-to-end."""
from collections import defaultdict

import numpy as np

from poultryai.data.schema import (DISEASES, ENV_CHANNELS, N_DISEASES, VISION_CHANNELS, Window)
from poultryai.eval.leadtime import evaluate_leadtime, tune_threshold_for_fpr
from poultryai.tasks.fusion.synth import SynthConfig, generate_dataset, generate_flock
from poultryai.tasks.fusion.leadtime_io import (build_flock_series, flock_key,
                                                signature_proxy_scores)

# small, fallback-env (no twin dependency) config for fast deterministic tests
CFG = SynthConfig(n_flocks=8, flock_days=2.0, seed=3, use_twin=False)


def _dataset():
    return generate_dataset(CFG)


def test_windows_are_schema_valid():
    W = _dataset()
    assert len(W) > 0
    for w in W[:20]:
        assert isinstance(w, Window)
        assert w.env.shape == (CFG.window_minutes, len(ENV_CHANNELS))
        assert w.vision.ndim == 2 and w.vision.shape[1] == len(VISION_CHANNELS)
        assert w.audio.ndim == 1 and w.audio.shape[0] == int(CFG.audio_seconds * CFG.sample_rate)
        assert w.labels.shape == (N_DISEASES,)
        assert w.onset_offset.shape == (N_DISEASES,)


def test_healthy_flocks_have_no_labels_or_onset():
    W = _dataset()
    by_flock = defaultdict(list)
    for w in W:
        by_flock[flock_key(w)].append(w)
    for ws in by_flock.values():
        diseased = np.isfinite(np.stack([w.onset_offset for w in ws])).any()
        if not diseased:
            assert all(not w.labels.any() for w in ws)          # healthy → all-zero labels


def test_onset_offset_sign_and_labels_within_horizon():
    # a single diseased flock: offset decreases with time; positive labels only within horizon
    rng = np.random.default_rng(0)
    ws = generate_flock(SynthConfig(flock_days=3.0, use_twin=False), 0, DISEASES[2], rng)
    d = DISEASES.index(DISEASES[2])
    offs = np.array([w.onset_offset[d] for w in ws])
    assert np.isfinite(offs).all()                              # diseased → finite for its disease
    assert (np.diff([w.t_end for w in ws]) > 0).all()          # windows ordered in time
    # every positive-labelled window is within the labelling horizon before/after onset
    horizon_s = SynthConfig().label_horizon_h * 3600.0
    for w in ws:
        if w.labels[d] == 1.0:
            assert w.onset_offset[d] <= horizon_s + 1


def test_flock_grouped_series_and_leadtime_pipeline():
    W = _dataset()
    scores = signature_proxy_scores(W)          # oracle proxy — validates plumbing only
    got_result = False
    for d in range(N_DISEASES):
        fs = build_flock_series(W, scores, d)
        assert len(fs) == len({flock_key(w) for w in W})        # one series per flock
        if not any(f.is_diseased for f in fs):
            continue
        thr = tune_threshold_for_fpr(fs, 0.05)
        r = evaluate_leadtime(fs, thr)
        if r.n_diseased:
            got_result = True
            assert r.detected_before_onset >= 0.5               # proxy should detect most
            assert 0.0 <= r.flock_false_positive_rate <= 1.0
    assert got_result


def test_signature_proxy_zero_for_healthy():
    W = _dataset()
    scores = signature_proxy_scores(W)
    for i, w in enumerate(W):
        if not np.isfinite(w.onset_offset).any():
            assert np.allclose(scores[i], 0.0)
