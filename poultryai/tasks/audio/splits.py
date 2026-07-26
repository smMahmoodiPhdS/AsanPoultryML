"""Scan the Mendeley audio corpus and build a recording-level stratified split (torch-free).

Uses only the ``wave`` stdlib module to read duration/sample-rate, so the manifest builds
without torch/torchaudio. Splits at the **recording** level (stratified by class) — windows
from one recording therefore never straddle train/val/test.
"""
from __future__ import annotations

import wave
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import FOLDER_TO_CLASS


@dataclass(slots=True)
class AudioRecord:
    path: str
    label: str
    duration_s: float
    sample_rate: int
    split: str = ""


def probe_wav(path: str | Path) -> tuple[float, int]:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate(), w.getframerate()


def scan_corpus(root: str | Path) -> list[AudioRecord]:
    """Scan ``{root}/{Healthy,Unhealthy,Noise}/*.wav`` into records."""
    root = Path(root)
    recs: list[AudioRecord] = []
    for folder, cls in FOLDER_TO_CLASS.items():
        for p in sorted((root / folder).glob("*.wav")):
            try:
                dur, sr = probe_wav(p)
            except Exception:
                continue
            recs.append(AudioRecord(path=str(p), label=cls, duration_s=dur, sample_rate=sr))
    return recs


def stratified_split(recs: list[AudioRecord], val_frac: float = 0.15,
                     test_frac: float = 0.15, seed: int = 42) -> dict[str, int]:
    """Assign each recording to train/val/test, stratified by class. Mutates in place."""
    rng = np.random.default_rng(seed)
    by_cls: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(recs):
        by_cls[r.label].append(i)
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for idxs in by_cls.values():
        idxs = list(idxs); rng.shuffle(idxs)
        n = len(idxs)
        n_test = max(1, round(n * test_frac)) if n > 2 else 0
        n_val = max(1, round(n * val_frac)) if n > 2 else 0
        for k, i in enumerate(idxs):
            split = "test" if k < n_test else "val" if k < n_test + n_val else "train"
            recs[i].split = split
            counts[split] += 1
    return counts


def assert_recording_level(recs: list[AudioRecord]) -> None:
    """Each path appears once with a single split (trivially true, but guards regressions)."""
    seen: dict[str, str] = {}
    for r in recs:
        s = seen.setdefault(r.path, r.split)
        if s != r.split:
            raise AssertionError(f"recording {r.path} spans splits {s} and {r.split}")
