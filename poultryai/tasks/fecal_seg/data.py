"""Manifest-driven segmentation dataset (needs torch/torchvision).

Serves (image, mask) pairs for a split from the V2 manifest (which inherited V1's
leakage-safe split). Images/masks are resized to a fixed square; masks use nearest-neighbour
so class indices are preserved. Light geometric+photometric augmentation on train, applied
identically to image and mask for the geometric parts.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import N_SEG_CLASSES
from .masks import load_example
from poultryai.tasks.fecal.data import IMAGENET_MEAN, IMAGENET_STD


@dataclass(slots=True)
class SegRow:
    json_path: str
    basename: str
    label_idx: int
    split: str


def read_seg_manifest(manifest: str | Path, split: str | None = None) -> list[SegRow]:
    rows: list[SegRow] = []
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            if split is not None and r["split"] != split:
                continue
            rows.append(SegRow(r["json_path"], r["basename"], int(r["label_idx"]), r["split"]))
    if not rows:
        raise ValueError(f"no rows for split={split!r} in {manifest}")
    return rows


def seg_class_pixel_weights(manifest: str | Path, split: str = "train",
                            sample: int = 300, seed: int = 42) -> np.ndarray:
    """Approximate inverse-frequency pixel weights over SEG classes (for weighted CE).

    Estimated from a random sample of ``sample`` masks (decoding every mask would be slow);
    background dominates, so this down-weights it and up-weights the rare disease pixels.
    """
    rows = read_seg_manifest(manifest, split)
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(rows), size=min(sample, len(rows)), replace=False)
    counts = np.zeros(N_SEG_CLASSES, dtype=np.float64)
    for i in idxs:
        _, mask = load_example(rows[i].json_path)
        c = np.bincount(mask.reshape(-1), minlength=N_SEG_CLASSES)
        counts += c[:N_SEG_CLASSES]
    counts[counts == 0] = 1.0
    w = counts.sum() / (N_SEG_CLASSES * counts)
    return (w / w.mean()).astype(np.float32)


class FecalSegDataset:
    """torch ``Dataset`` over a seg-manifest split (wrap via :func:`make_torch_seg_dataset`)."""

    def __init__(self, manifest: str | Path, split: str, img_size: int = 512,
                 train: bool | None = None):
        from torch.utils.data import Dataset  # noqa: F401 — ensure torch present
        self.rows = read_seg_manifest(manifest, split)
        self.img_size = img_size
        self._train = (split == "train") if train is None else train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        import torch
        from PIL import Image
        import torchvision.transforms.functional as TF

        img, mask = load_example(self.rows[i].json_path)          # PIL RGB, np.uint8 HxW
        mask_img = Image.fromarray(mask)
        S = self.img_size
        img = img.resize((S, S), Image.BILINEAR)
        mask_img = mask_img.resize((S, S), Image.NEAREST)

        if self._train:
            if np.random.rand() < 0.5:
                img = TF.hflip(img); mask_img = TF.hflip(mask_img)
            if np.random.rand() < 0.5:
                img = TF.vflip(img); mask_img = TF.vflip(mask_img)
            img = TF.adjust_brightness(img, 0.8 + 0.4 * np.random.rand())
            img = TF.adjust_saturation(img, 0.8 + 0.4 * np.random.rand())

        x = TF.to_tensor(img)
        x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        y = torch.from_numpy(np.asarray(mask_img, dtype=np.int64))
        return x, y


def make_torch_seg_dataset(*args, **kwargs):
    from torch.utils.data import Dataset

    class _TorchSeg(Dataset, FecalSegDataset):
        pass

    return _TorchSeg(*args, **kwargs)
