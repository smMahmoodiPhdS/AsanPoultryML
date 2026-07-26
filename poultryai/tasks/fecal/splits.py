"""Leakage-safe manifest construction for the fecal classifier (torch-free).

Why this module exists
----------------------
The Kaggle and Zenodo fecal datasets are two representations of the *same* source corpus,
and within a single dataset the same fecal sample can appear as several near-identical
photos. If those duplicates straddle the train/test boundary, test accuracy is inflated —
the classic livestock-CV mistake flagged in ``docs/methodology.md`` and ``ai-model-plan.md``.

Guard (two layers)
------------------
1. **Exact dedup** by MD5 of the file bytes — identical files are collapsed to one row.
2. **Near-duplicate grouping** by a 64-bit **average hash** (aHash). Images whose aHash is
   within a small Hamming radius are placed in the same *group*, and whole groups are
   assigned to a single split. This keeps re-crops/re-compressions of one sample together.

The split is **stratified by class** at the group level so every split preserves the class
mix despite the strong imbalance (Newcastle is the minority).

This module depends only on the standard library + numpy so it can be unit-tested and run
without a torch install.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import PREFIX_TO_CLASS


# --------------------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------------------
def md5_of_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Exact content hash of a file (streamed, so large files are fine)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


_DCT_BASIS: dict[int, np.ndarray] = {}


def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II basis (cached); avoids a scipy dependency."""
    if n not in _DCT_BASIS:
        k = np.arange(n)
        m = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
        m[0, :] *= 1 / np.sqrt(2)
        _DCT_BASIS[n] = m * np.sqrt(2 / n)
    return _DCT_BASIS[n]


def phash(path: str | Path, img_size: int = 32, hash_size: int = 8) -> int:
    """64-bit DCT perceptual hash (pHash).

    pHash keeps the low-frequency *structure* of the image, so — unlike the coarse
    average-hash — it does **not** collapse distinct fecal photos that merely share a
    background/lighting. Uses PIL ``draft`` fast-decode; returns 0 on read failure so the
    caller falls back to md5-only for that file.
    """
    try:
        from PIL import Image
    except Exception:  # pragma: no cover - PIL always present in this project
        return 0
    try:
        with Image.open(path) as im:
            im.draft("L", (img_size * 2, img_size * 2))
            im = im.convert("L").resize((img_size, img_size))
            px = np.asarray(im, dtype=np.float32)
    except Exception:
        return 0
    d = _dct_matrix(img_size)
    coef = d @ px @ d.T                          # 2-D DCT-II
    low = coef[:hash_size, :hash_size].flatten()
    bits = low > np.median(low[1:])              # drop the DC term from the threshold
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


# Backwards-compatible alias; the manifest builder uses ``phash`` by default now.
average_hash = phash


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------
@dataclass(slots=True)
class ImageRecord:
    path: str
    label: str
    is_pcr: bool
    md5: str
    ahash: int
    group_id: int = -1
    split: str = ""


def class_from_filename(name: str) -> tuple[str, bool]:
    """Map a Kaggle-style filename (e.g. ``pcrcocci.12.jpg``) to (class, is_pcr)."""
    stem = Path(name).name.split(".")[0].lower()
    cls = PREFIX_TO_CLASS.get(stem)
    if cls is None:  # some files may use the un-prefixed disease token only
        for pfx, c in PREFIX_TO_CLASS.items():
            if stem.startswith(pfx):
                cls = c
                break
    if cls is None:
        raise ValueError(f"cannot map filename to class: {name!r}")
    return cls, stem.startswith("pcr")


# --------------------------------------------------------------------------------------
# Grouping (near-duplicate union-find over aHash)
# --------------------------------------------------------------------------------------
def assign_groups(records: list[ImageRecord], max_hamming: int = 6) -> int:
    """Union near-duplicate images into groups (in place). Returns the group count.

    Two layers:
    1. **Exact md5 duplicates** are always unioned.
    2. **pHash near-duplicates within the *same class*** and within ``max_hamming`` bits
       are unioned. Restricting to same-class is deliberate: fecal photos share
       backgrounds, so a *cross-class* hash neighbour is a collision, not a real
       duplicate — merging it would corrupt the split (an early aHash version produced an
       879-image cluster spanning all four classes; pHash + same-class restriction fixes
       it). ``max_hamming=0`` disables near-dup grouping (md5-only).

    Comparison is O(n^2) *within each class* only (≤ a few thousand each), which is cheap.
    """
    n = len(records)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # 1) exact md5 duplicates (any class — identical bytes are identical files)
    by_md5: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        by_md5[r.md5].append(i)
    for idxs in by_md5.values():
        for j in idxs[1:]:
            union(idxs[0], j)

    # 2) pHash near-duplicates, compared only within the same class
    if max_hamming > 0:
        by_class: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(records):
            by_class[r.label].append(i)
        for idxs in by_class.values():
            for a in range(len(idxs)):
                ia = idxs[a]
                ha = records[ia].ahash
                if ha == 0:
                    continue
                for b in range(a + 1, len(idxs)):
                    ib = idxs[b]
                    hb = records[ib].ahash
                    if hb and _hamming(ha, hb) <= max_hamming:
                        union(ia, ib)

    roots: dict[int, int] = {}
    for i in range(n):
        r = find(i)
        gid = roots.setdefault(r, len(roots))
        records[i].group_id = gid
    return len(roots)


# --------------------------------------------------------------------------------------
# Stratified group split
# --------------------------------------------------------------------------------------
def stratified_group_split(
    records: list[ImageRecord],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, int]:
    """Assign each *group* to train/val/test, stratified by the group's majority class.

    Assigns whole groups (never splitting a near-dup cluster), and balances each class's
    groups across splits so the class proportions are preserved. Returns per-split counts.
    """
    rng = np.random.default_rng(seed)

    # group -> member indices, and group -> majority class
    members: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        members[r.group_id].append(i)
    group_class: dict[int, str] = {}
    for gid, idxs in members.items():
        labels = [records[i].label for i in idxs]
        group_class[gid] = max(set(labels), key=labels.count)

    # stratify: for each class, shuffle its groups and slice by fraction
    groups_by_class: dict[str, list[int]] = defaultdict(list)
    for gid, cls in group_class.items():
        groups_by_class[cls].append(gid)

    split_of_group: dict[int, str] = {}
    for cls, gids in groups_by_class.items():
        gids = list(gids)
        rng.shuffle(gids)
        n = len(gids)
        n_test = max(1, int(round(n * test_frac))) if n > 2 else 0
        n_val = max(1, int(round(n * val_frac))) if n > 2 else 0
        for k, gid in enumerate(gids):
            if k < n_test:
                split_of_group[gid] = "test"
            elif k < n_test + n_val:
                split_of_group[gid] = "val"
            else:
                split_of_group[gid] = "train"

    counts: dict[str, int] = defaultdict(int)
    for r in records:
        r.split = split_of_group[r.group_id]
        counts[r.split] += 1
    return dict(counts)


def assert_no_leakage(records: list[ImageRecord]) -> None:
    """Raise if any group_id (or md5) appears in more than one split — the safety check."""
    for key in ("group_id", "md5"):
        seen: dict[object, str] = {}
        for r in records:
            k = getattr(r, key)
            s = seen.setdefault(k, r.split)
            if s != r.split:
                raise AssertionError(f"leakage: {key}={k!r} spans splits {s} and {r.split}")
