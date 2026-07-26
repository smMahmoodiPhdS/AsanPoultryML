"""Manifest-driven dataset + transforms for the fecal classifier (needs torch/torchvision).

Reads the CSV written by ``scripts/build_fecal_manifest.py`` (the leakage-safe split) and
serves images for a requested split. Kept deliberately thin: the scientific integrity work
(dedup, group-aware split) already happened in ``splits.py`` and is frozen in the manifest.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import CLASS_TO_IDX, CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(slots=True)
class Row:
    path: str
    label_idx: int
    is_pcr: bool
    split: str


def read_manifest(manifest: str | Path, split: str | None = None) -> list[Row]:
    rows: list[Row] = []
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            if split is not None and r["split"] != split:
                continue
            rows.append(Row(path=r["path"], label_idx=int(r["label_idx"]),
                            is_pcr=bool(int(r["is_pcr"])), split=r["split"]))
    if not rows:
        raise ValueError(f"no rows for split={split!r} in {manifest}")
    return rows


def class_weights(rows: list[Row], n_classes: int = len(CLASSES)) -> np.ndarray:
    """Inverse-frequency class weights (for the imbalanced CE loss)."""
    counts = np.bincount([r.label_idx for r in rows], minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return (w / w.mean()).astype(np.float32)


def build_transforms(img_size: int = 224, train: bool = True):
    """torchvision transform pipeline. Imported lazily so the module loads without torch."""
    from torchvision import transforms as T
    if train:
        return T.Compose([
            T.Resize((img_size + 32, img_size + 32)),
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(0.2, 0.2, 0.2, 0.05),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class FecalDataset:
    """torch ``Dataset`` over a manifest split. (Base class is torch's ``Dataset``.)"""

    def __init__(self, manifest: str | Path, split: str, img_size: int = 224,
                 train: bool | None = None, pcr_only: bool = False):
        from torch.utils.data import Dataset  # noqa: F401 — ensure torch present
        self.rows = read_manifest(manifest, split)
        if pcr_only:
            self.rows = [r for r in self.rows if r.is_pcr]
        self._train = (split == "train") if train is None else train
        self.tf = build_transforms(img_size, train=self._train)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        from PIL import Image
        r = self.rows[i]
        with Image.open(r.path) as im:
            img = self.tf(im.convert("RGB"))
        return img, r.label_idx


def make_torch_dataset(*args, **kwargs):
    """Return a real ``torch.utils.data.Dataset`` wrapping :class:`FecalDataset`.

    Kept separate so ``FecalDataset`` can be imported/inspected without subclassing torch
    at module-import time.
    """
    from torch.utils.data import Dataset

    class _TorchFecal(Dataset, FecalDataset):
        pass

    return _TorchFecal(*args, **kwargs)
