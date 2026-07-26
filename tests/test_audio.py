"""Torch-free tests for A1: windowing, recording-level split, and eval aggregation."""
import csv
from pathlib import Path

import numpy as np

from poultryai.tasks.audio import CLASS_TO_IDX
from poultryai.tasks.audio.windowing import window_offsets, n_windows, pad_or_trim
from poultryai.tasks.audio.eval import window_report, recording_report


# ---- windowing ----
def test_short_clip_gives_one_window():
    assert window_offsets(1.0, window_s=4.0, sample_rate=16000) == [0]      # 1s < 4s
    assert n_windows(0.5, 4.0, sample_rate=16000) == 1


def test_nonoverlap_window_count():
    # 12s clip, 4s non-overlapping windows → 3 windows at 0,4,8 s
    offs = window_offsets(12.0, window_s=4.0, hop_s=4.0, sample_rate=16000)
    assert offs == [0, 64000, 128000]


def test_overlap_hop_more_windows():
    non = n_windows(12.0, 4.0, hop_s=4.0, sample_rate=16000)
    ovl = n_windows(12.0, 4.0, hop_s=2.0, sample_rate=16000)
    assert ovl > non


def test_max_windows_cap():
    capped = window_offsets(600.0, 4.0, hop_s=4.0, max_windows=20, sample_rate=16000)
    assert len(capped) <= 20


def test_pad_or_trim():
    assert len(pad_or_trim(np.ones(10), 16)) == 16
    assert len(pad_or_trim(np.ones(20), 16)) == 16
    padded = pad_or_trim(np.ones(4), 8)
    assert padded[:4].sum() == 4 and padded[4:].sum() == 0


# ---- eval aggregation ----
def test_recording_aggregation_flips_to_true_label():
    # two noisy windows per recording; their mean points to the true class
    probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.7, 0.1],      # rec A (true 0): mean→0.4,0.5,0.1?
                      [0.9, 0.05, 0.05], [0.8, 0.1, 0.1]])    # rec B (true 0)
    labels = np.array([0, 0, 0, 0])
    ids = ["A", "A", "B", "B"]
    rep = recording_report(probs, labels, ids)
    assert rep["n_recordings"] == 2
    assert 0.0 <= rep["accuracy"] <= 1.0


def test_window_report_keys():
    probs = np.eye(3)[np.array([0, 1, 2, 0])]
    rep = window_report(probs * 0.7 + 0.1, np.array([0, 1, 2, 0]))
    assert rep["n"] == 4 and "ece" in rep and rep["accuracy"] == 1.0


# ---- manifest integrity (if built) ----
def test_manifest_recording_level_split():
    mp = Path("poultryai/tasks/audio/manifest.csv")
    if not mp.exists():
        return
    rows = list(csv.DictReader(open(mp)))
    split_of = {}
    for r in rows:
        split_of.setdefault(r["path"], set()).add(r["split"])
    assert all(len(s) == 1 for s in split_of.values())
    # all three classes present in each split
    for sp in ("train", "val", "test"):
        labs = {r["label"] for r in rows if r["split"] == sp}
        assert labs == set(CLASS_TO_IDX)
