#!/usr/bin/env python3
"""Build the leakage-safe manifest for the fecal disease classifier (V1).

Scans the Kaggle fecal images, computes an exact hash (dedup) and a perceptual hash
(near-duplicate grouping), maps each file to a canonical class, groups near-duplicates,
and writes a stratified train/val/test split to a CSV manifest that ``train_fecal.py``
consumes. This is the **leakage gate**: no downstream training is valid without it.

    python scripts/build_fecal_manifest.py \
        --images Data/downloads/Kaggle/kaggle_chicken_disease/Train \
        --out    poultryai/tasks/fecal/manifest.csv

Torch is *not* required — only numpy + PIL. Run once; commit the manifest for provenance.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poultryai.tasks.fecal import CLASS_TO_IDX  # noqa: E402
from poultryai.tasks.fecal.splits import (  # noqa: E402
    ImageRecord, assign_groups, assert_no_leakage, average_hash,
    class_from_filename, md5_of_file, stratified_group_split,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def drop_label_conflicts(records: list[ImageRecord]) -> list[ImageRecord]:
    """Drop byte-identical images that carry *conflicting* class labels.

    The source corpus contains a few files that are md5-identical yet filed under two
    different disease prefixes (e.g. ``pcrcocci.228.jpg`` == ``pcrhealthy.83.jpg``). Their
    true label is unknowable, so keeping either would inject label noise; we drop the whole
    conflicting set and report it for provenance.
    """
    from collections import defaultdict
    labels_by_md5: dict[str, set[str]] = defaultdict(set)
    for r in records:
        labels_by_md5[r.md5].add(r.label)
    conflicted = {m for m, ls in labels_by_md5.items() if len(ls) > 1}
    if conflicted:
        print(f"dropping {sum(r.md5 in conflicted for r in records)} images across "
              f"{len(conflicted)} md5-identical-but-conflicting-label sets")
    return [r for r in records if r.md5 not in conflicted]


def scan(images_dir: Path, limit: int | None = None) -> list[ImageRecord]:
    files = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if limit:
        files = files[:limit]
    records: list[ImageRecord] = []
    t0 = time.time()
    skipped = 0
    for i, p in enumerate(files):
        try:
            cls, is_pcr = class_from_filename(p.name)
        except ValueError:
            skipped += 1
            continue
        records.append(ImageRecord(
            path=str(p), label=cls, is_pcr=is_pcr,
            md5=md5_of_file(p), ahash=average_hash(p),
        ))
        if (i + 1) % 1000 == 0:
            print(f"  hashed {i + 1}/{len(files)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"scanned {len(records)} images ({skipped} skipped, unmapped names)")
    return records


def write_manifest(records: list[ImageRecord], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "label_idx", "is_pcr", "md5", "ahash",
                    "group_id", "split"])
        for r in records:
            w.writerow([r.path, r.label, CLASS_TO_IDX[r.label], int(r.is_pcr),
                        r.md5, r.ahash, r.group_id, r.split])


def summarise(records: list[ImageRecord]) -> None:
    from collections import Counter
    print("\n=== manifest summary ===")
    n = len(records)
    n_groups = len({r.group_id for r in records})
    print(f"images: {n}  |  unique groups (after dedup): {n_groups}  "
          f"|  exact-dup rows collapsed into groups: {n - n_groups}")
    for split in ("train", "val", "test"):
        rows = [r for r in records if r.split == split]
        by_cls = Counter(r.label for r in rows)
        pcr = sum(r.is_pcr for r in rows)
        dist = "  ".join(f"{c}:{by_cls.get(c, 0)}" for c in CLASS_TO_IDX)
        print(f"  {split:5s} n={len(rows):5d}  ({dist})  pcr={pcr}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="root dir of fecal JPGs")
    ap.add_argument("--out", default="poultryai/tasks/fecal/manifest.csv")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--max-hamming", type=int, default=0,
                    help="pHash Hamming radius for near-dup grouping. Default 0 = exact "
                         "md5 dedup only (recommended: on this corpus no perceptual "
                         "threshold is stable — even radius 1 collapses same-class photos, "
                         "so pHash is stored as a review flag, not used to force groups).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="debug: cap #images")
    args = ap.parse_args()

    records = scan(Path(args.images), limit=args.limit)
    if not records:
        raise SystemExit("no images found — check --images path")
    records = drop_label_conflicts(records)
    n_groups = assign_groups(records, max_hamming=args.max_hamming)
    print(f"grouped into {n_groups} near-duplicate clusters "
          f"(max_hamming={args.max_hamming})")
    counts = stratified_group_split(records, args.val_frac, args.test_frac, args.seed)
    assert_no_leakage(records)          # hard gate
    print(f"split counts: {counts}  (leakage check passed)")
    write_manifest(records, Path(args.out))
    summarise(records)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
