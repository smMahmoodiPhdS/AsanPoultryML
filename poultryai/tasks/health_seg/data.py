"""Health-seg dataset — reads the V3 clip-aware manifest (needs torch/torchvision).

The manifest ``poultryai/tasks/vision/manifests/roboflow_healthy_sick.csv`` was built by V3's
``build_roboflow_manifest.py`` with a **clip-grouped, leakage-free** split. V4 reuses it
directly (columns: ``path, label_path, clip, orig_split, split, ...``), so no new split logic
is needed and the leakage guard is shared with V3.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import N_SEG_CLASSES
from .masks import label_pixel_hist, load_example
from poultryai.tasks.fecal.data import IMAGENET_MEAN, IMAGENET_STD


@dataclass(slots=True)
class SegRow:
    path: str
    label_path: str
    clip: str
    split: str


def read_manifest(manifest: str | Path, split: str | None = None) -> list[SegRow]:
    rows: list[SegRow] = []
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            if split is not None and r["split"] != split:
                continue
            rows.append(SegRow(r["path"], r["label_path"], r["clip"], r["split"]))
    if not rows:
        raise ValueError(f"no rows for split={split!r} in {manifest}")
    return rows


def seg_class_pixel_weights(manifest: str | Path, split: str = "train",
                            img_size: int = 416, sample: int = 200, seed: int = 42) -> np.ndarray:
    """Inverse-frequency pixel weights over {background, healthy, sick} from a sample."""
    rows = read_manifest(manifest, split)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(rows), size=min(sample, len(rows)), replace=False)
    counts = np.zeros(N_SEG_CLASSES, dtype=np.float64)
    for i in idxs:
        counts += label_pixel_hist(rows[i].label_path, img_size, img_size)
    counts[counts == 0] = 1.0
    w = counts.sum() / (N_SEG_CLASSES * counts)
    return (w / w.mean()).astype(np.float32)


class HealthSegDataset:
    """torch ``Dataset`` (wrap via :func:`make_torch_seg_dataset`)."""

    def __init__(self, manifest: str | Path, split: str, img_size: int = 416,
                 train: bool | None = None):
        from torch.utils.data import Dataset  # noqa: F401
        self.rows = read_manifest(manifest, split)
        self.img_size = img_size
        self._train = (split == "train") if train is None else train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        import torch
        from PIL import Image
        import torchvision.transforms.functional as TF

        img, mask = load_example(self.rows[i].path, self.rows[i].label_path)
        mask_img = Image.fromarray(mask)
        S = self.img_size
        img = img.resize((S, S), Image.BILINEAR)
        mask_img = mask_img.resize((S, S), Image.NEAREST)
        if self._train:
            if np.random.rand() < 0.5:
                img = TF.hflip(img); mask_img = TF.hflip(mask_img)
            img = TF.adjust_brightness(img, 0.8 + 0.4 * np.random.rand())
            img = TF.adjust_contrast(img, 0.8 + 0.4 * np.random.rand())
        x = TF.normalize(TF.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)
        y = torch.from_numpy(np.asarray(mask_img, dtype=np.int64))
        return x, y


def make_torch_seg_dataset(*args, **kwargs):
    from torch.utils.data import Dataset

    class _Torch(Dataset, HealthSegDataset):
        pass

    return _Torch(*args, **kwargs)
