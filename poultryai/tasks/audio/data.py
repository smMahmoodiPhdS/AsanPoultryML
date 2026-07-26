"""Audio dataset: recording manifest → fixed 16 kHz waveform windows (needs torch/torchaudio).

Each manifest recording is expanded into window items (path, start-sample, label). Train uses
an overlapping hop + per-file cap (augmentation + balance); eval uses non-overlapping windows.
Waveforms are loaded, downmixed to mono, resampled 48 kHz→16 kHz, and padded/trimmed to exactly
``window_s`` — the raw waveform the in-model ``LogMelFrontend`` consumes (so ONNX needs no
external preprocessing). A soundfile loader is used if torchaudio is unavailable.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import CLASSES
from .windowing import pad_or_trim, window_offsets


@dataclass(slots=True)
class WindowItem:
    path: str
    start: int          # start sample at target sr
    label_idx: int


def _load_wav_16k(path: str, target_sr: int) -> np.ndarray:
    """Load a wav → mono float32 at ``target_sr``. Prefers torchaudio, falls back to soundfile."""
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)                 # (C, N) float32 in [-1,1]
        wav = wav.mean(0, keepdim=True)                 # mono
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav.squeeze(0).numpy()
    except Exception:
        import soundfile as sf
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = wav.mean(1)
        if sr != target_sr:                             # simple linear resample fallback
            n = int(round(len(wav) * target_sr / sr))
            wav = np.interp(np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav)
        return wav.astype(np.float32)


def build_window_items(manifest: str | Path, split: str, *, window_s: float, hop_s: float,
                       max_windows: int, sample_rate: int, eval_mode: bool) -> list[WindowItem]:
    items: list[WindowItem] = []
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            if r["split"] != split:
                continue
            hop = window_s if eval_mode else hop_s      # non-overlap at eval
            cap = None if eval_mode else max_windows
            for st in window_offsets(float(r["duration_s"]), window_s, hop, cap, sample_rate):
                items.append(WindowItem(r["path"], st, int(r["label_idx"])))
    if not items:
        raise ValueError(f"no window items for split={split!r} in {manifest}")
    return items


class AudioWindows:
    """torch ``Dataset`` of waveform windows (wrap via :func:`make_torch_audio_dataset`)."""

    def __init__(self, manifest: str | Path, split: str, window_s: float = 4.0,
                 hop_s: float = 2.0, max_windows: int = 20, sample_rate: int = 16_000,
                 eval_mode: bool | None = None):
        from torch.utils.data import Dataset  # noqa: F401
        self.sr = sample_rate
        self.win = int(round(window_s * sample_rate))
        self.eval_mode = (split != "train") if eval_mode is None else eval_mode
        self.items = build_window_items(manifest, split, window_s=window_s, hop_s=hop_s,
                                        max_windows=max_windows, sample_rate=sample_rate,
                                        eval_mode=self.eval_mode)
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.items)

    def _wav(self, path: str) -> np.ndarray:
        if path not in self._cache:
            if len(self._cache) > 64:                   # bounded cache
                self._cache.pop(next(iter(self._cache)))
            self._cache[path] = _load_wav_16k(path, self.sr)
        return self._cache[path]

    def __getitem__(self, i: int):
        import torch
        it = self.items[i]
        wav = self._wav(it.path)
        seg = pad_or_trim(wav[it.start: it.start + self.win], self.win)
        return torch.from_numpy(np.ascontiguousarray(seg, dtype=np.float32)), it.label_idx


def make_torch_audio_dataset(*args, **kwargs):
    from torch.utils.data import Dataset

    class _Torch(Dataset, AudioWindows):
        pass

    return _Torch(*args, **kwargs)


def window_class_weights(manifest: str | Path, split: str = "train", **kw) -> np.ndarray:
    items = build_window_items(manifest, split, window_s=kw.get("window_s", 4.0),
                               hop_s=kw.get("hop_s", 2.0), max_windows=kw.get("max_windows", 20),
                               sample_rate=kw.get("sample_rate", 16_000), eval_mode=False)
    counts = np.bincount([it.label_idx for it in items], minlength=len(CLASSES)).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (len(CLASSES) * counts)
    return (w / w.mean()).astype(np.float32)
