"""Dataset + DataModule for the aligned multimodal windows.

Key correctness concerns handled here:
* **Grouped, time-aware splitting** — split by *flock*, never by random window, to avoid
  temporal leakage (windows from one flock must not straddle train/val/test).
* **Per-channel normalisation fit on train only** and applied to val/test.
* **Modality availability mask** so the model learns to cope with dropouts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import ENV_CHANNELS, VISION_CHANNELS, Window


class WindowDataset(Dataset):
    def __init__(self, windows: list[Window], env_mean=None, env_std=None) -> None:
        self.windows = windows
        self.env_mean = env_mean
        self.env_std = env_std

    def fit_normalisation(self) -> tuple[np.ndarray, np.ndarray]:
        stack = np.concatenate([w.env for w in self.windows], axis=0)
        self.env_mean = stack.mean(0); self.env_std = stack.std(0) + 1e-6
        return self.env_mean, self.env_std

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> dict:
        w = self.windows[i]
        env = w.env
        if self.env_mean is not None:
            env = (env - self.env_mean) / self.env_std
        present = torch.tensor(
            [np.isfinite(env).all(), w.audio.size > 0, np.isfinite(w.vision).all()],
            dtype=torch.bool)
        return {
            "env": torch.as_tensor(np.nan_to_num(env), dtype=torch.float32),
            "audio": torch.as_tensor(w.audio, dtype=torch.float32),
            "vision": torch.as_tensor(np.nan_to_num(w.vision), dtype=torch.float32),
            "meta": torch.as_tensor(w.meta.to_vector(), dtype=torch.float32),
            "labels": torch.as_tensor(w.labels, dtype=torch.float32),
            "present": present,
        }


def split_by_flock(windows: list[Window], val_frac=0.15, test_frac=0.15, seed=42):
    """Group-holdout split keyed on (owner, saloon) to prevent flock leakage."""
    rng = np.random.default_rng(seed)
    groups = sorted({(w.meta.owner, w.meta.saloon) for w in windows})
    rng.shuffle(groups)
    n = len(groups)
    n_test = max(1, int(round(test_frac * n)))
    n_val = max(1, int(round(val_frac * n)))
    test_g = set(groups[:n_test])
    val_g = set(groups[n_test:n_test + n_val])
    train, val, test = [], [], []
    for w in windows:
        g = (w.meta.owner, w.meta.saloon)
        (test if g in test_g else val if g in val_g else train).append(w)
    return train, val, test


def collate(batch: list[dict]) -> dict:
    # audio waveforms may differ in length -> pad to the longest in the batch
    max_len = max(b["audio"].shape[0] for b in batch)
    out: dict = {}
    for k in batch[0]:
        if k == "audio":
            out[k] = torch.stack([
                torch.nn.functional.pad(b[k], (0, max_len - b[k].shape[0])) for b in batch])
        else:
            out[k] = torch.stack([b[k] for b in batch])
    return out
