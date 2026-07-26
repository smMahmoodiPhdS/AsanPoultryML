"""YOLO dataset scanning + clip-aware split & leakage audit (torch-free).

Why this exists
---------------
Roboflow ships train/valid/test folders, but its default split is *random over frames*.
When a dataset is built from video clips (this project's tracking/detection sets are), frames
from one clip end up in several splits — near-duplicate frames that inflate detection and
tracking scores. This module recovers the **source clip id** from each filename and provides a
**clip-grouped** re-split so no clip spans two splits, plus an audit of the shipped split.

Filename convention (Roboflow): ``<original>_<ext>.rf.<hash>.<ext>`` where ``<original>`` is
the source frame name, often ``<clip>...<frameindex>``. We strip the ``.rf.<hash>`` suffix, the
``_jpg``/``_png`` token, and a trailing frame index to obtain the clip id.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def clip_id(filename: str) -> str:
    """Recover a source-clip id from a Roboflow filename (robust to hash/ext variants)."""
    name = Path(filename).name
    base = name.split(".rf.")[0]                       # drop ".rf.<hash>.<ext>"
    base = re.sub(r"_(jpg|jpeg|png|bmp)$", "", base, flags=re.I)   # drop "_jpg"/"_png" token
    # drop a trailing frame index like  _mp4-12 | -0007 | _34 | 34
    stripped = re.sub(r"([_-]mp4)?[-_]?\d+$", "", base)
    # if the name was *only* a frame index (e.g. "01"), keep it so its augmentations still
    # group together (falling back to the full hashed name would split same-frame duplicates)
    return stripped or base


@dataclass(slots=True)
class YoloImage:
    dataset: str
    path: str
    label_path: str
    orig_split: str          # split as shipped by Roboflow
    clip: str
    n_objects: int
    class_hist: tuple[int, ...]
    split: str = ""          # clip-aware split assigned here


def _read_label(label_path: Path, n_classes: int) -> tuple[int, tuple[int, ...]]:
    hist = [0] * max(n_classes, 1)
    n = 0
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            cid = int(float(line.split()[0]))
            if cid < len(hist):
                hist[cid] += 1
            n += 1
    return n, tuple(hist)


def parse_data_yaml_names(dataset_dir: Path) -> list[str]:
    """Minimal parse of the ``names:`` list from data.yaml (avoids a yaml dep here)."""
    y = dataset_dir / "data.yaml"
    names: list[str] = []
    if y.exists():
        capture = False
        for line in y.read_text().splitlines():
            if line.startswith("names:"):
                capture = True
                inline = line.split(":", 1)[1].strip()
                if inline.startswith("["):
                    return [s.strip().strip("'\"") for s in inline.strip("[]").split(",") if s.strip()]
                continue
            if capture:
                m = re.match(r"\s*-\s*(.+)", line)
                if m:
                    names.append(m.group(1).strip().strip("'\""))
                elif line and not line[0].isspace():
                    break
    return names


def scan_dataset(dataset_dir: Path, dataset: str) -> list[YoloImage]:
    names = parse_data_yaml_names(dataset_dir)
    n_classes = len(names) or 1
    imgs: list[YoloImage] = []
    for split in ("train", "valid", "test"):
        img_dir = dataset_dir / split / "images"
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.iterdir()):
            if p.suffix.lower() not in _IMG_EXTS:
                continue
            lp = dataset_dir / split / "labels" / (p.stem + ".txt")
            n, hist = _read_label(lp, n_classes)
            imgs.append(YoloImage(dataset=dataset, path=str(p), label_path=str(lp),
                                  orig_split=split, clip=clip_id(p.name),
                                  n_objects=n, class_hist=hist))
    return imgs


def audit_shipped_split(imgs: list[YoloImage]) -> dict[str, object]:
    """Count clips that span more than one *shipped* split (i.e. leaked)."""
    clip_splits: dict[str, set[str]] = defaultdict(set)
    for im in imgs:
        clip_splits[im.clip].add(im.orig_split)
    leaked = {c for c, s in clip_splits.items() if len(s) > 1}
    return {"n_images": len(imgs), "n_clips": len(clip_splits),
            "leaked_clips": len(leaked),
            "leaked_images": sum(im.n_objects >= 0 for im in imgs if im.clip in leaked)}


def clip_aware_split(imgs: list[YoloImage], val_frac: float = 0.15,
                     test_frac: float = 0.15, seed: int = 42) -> dict[str, int]:
    """Assign a leakage-free split: whole clips go to one split. Returns per-split counts.

    Stratified by each clip's dominant object class so class balance is preserved despite
    grouping. Mutates ``imgs`` in place (sets ``.split``).
    """
    rng = np.random.default_rng(seed)
    members: dict[str, list[int]] = defaultdict(list)
    for i, im in enumerate(imgs):
        members[im.clip].append(i)

    def clip_class(idxs: list[int]) -> int:
        agg = np.zeros(len(imgs[idxs[0]].class_hist) or 1)
        for i in idxs:
            h = imgs[i].class_hist
            if h:
                agg[: len(h)] += h
        return int(agg.argmax()) if agg.sum() > 0 else -1

    by_class: dict[int, list[str]] = defaultdict(list)
    for clip, idxs in members.items():
        by_class[clip_class(idxs)].append(clip)

    split_of: dict[str, str] = {}
    for clips in by_class.values():
        clips = list(clips); rng.shuffle(clips)
        n = len(clips)
        n_test = max(1, round(n * test_frac)) if n > 2 else 0
        n_val = max(1, round(n * val_frac)) if n > 2 else 0
        for k, c in enumerate(clips):
            split_of[c] = "test" if k < n_test else "val" if k < n_test + n_val else "train"

    counts: Counter[str] = Counter()
    for im in imgs:
        im.split = split_of[im.clip]
        counts[im.split] += 1
    return dict(counts)


def assert_clip_split_clean(imgs: list[YoloImage]) -> None:
    seen: dict[str, str] = {}
    for im in imgs:
        s = seen.setdefault(im.clip, im.split)
        if s != im.split:
            raise AssertionError(f"clip {im.clip!r} spans splits {s} and {im.split}")
