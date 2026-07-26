"""Clip → fixed-length window offsets (torch-free; unit-tested without torch).

Long recordings are cut into fixed ``window_s`` windows; short recordings are padded to one
window. A per-file cap prevents a single very long recording (the corpus has a 15-minute
Healthy clip) from dominating training. Windows never cross a recording boundary, so once a
*recording* is assigned to a split its windows inherit that split — the leakage guard.
"""
from __future__ import annotations

import numpy as np


def window_offsets(duration_s: float, window_s: float = 4.0, hop_s: float | None = None,
                   max_windows: int | None = None, sample_rate: int = 16_000) -> list[int]:
    """Return start-sample offsets for the windows of one clip.

    * ``hop_s`` None → non-overlapping (hop = window). Use a smaller hop for train-time overlap.
    * clips shorter than one window → a single window at offset 0 (the loader zero-pads).
    * ``max_windows`` caps the count (evenly subsampled across the clip) to balance long files.
    """
    sr = sample_rate
    win = int(round(window_s * sr))
    hop = int(round((hop_s if hop_s is not None else window_s) * sr))
    total = int(round(duration_s * sr))
    if total <= win:
        return [0]
    last_start = total - win
    starts = list(range(0, last_start + 1, max(hop, 1)))
    if not starts:
        starts = [0]
    if max_windows is not None and len(starts) > max_windows:
        idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = [starts[i] for i in sorted(set(idx.tolist()))]
    return starts


def n_windows(duration_s: float, window_s: float = 4.0, hop_s: float | None = None,
              max_windows: int | None = None, sample_rate: int = 16_000) -> int:
    return len(window_offsets(duration_s, window_s, hop_s, max_windows, sample_rate))


def pad_or_trim(wav: np.ndarray, length: int) -> np.ndarray:
    """Zero-pad (short) or trim (long) a 1-D waveform to exactly ``length`` samples."""
    wav = np.asarray(wav).reshape(-1)
    if len(wav) >= length:
        return wav[:length]
    out = np.zeros(length, dtype=wav.dtype if wav.dtype.kind == "f" else np.float32)
    out[: len(wav)] = wav
    return out
