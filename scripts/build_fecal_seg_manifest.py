#!/usr/bin/env python3
"""Build the segmentation manifest for V2, **inheriting V1's leakage-safe split**.

The Zenodo segmentation JSONs are the same source corpus as V1's Kaggle images, and their
``imagePath`` carries the identical filename (e.g. ``cocci.1819.jpg``). We therefore join
each JSON to V1's ``manifest.csv`` by filename and copy its train/val/test assignment, so
**no image V1 trained on is ever tested by V2** (the cross-model leakage guard). Any Zenodo
image with no V1 twin is reported and given a stratified fallback split.

    python scripts/build_fecal_seg_manifest.py \
        --zenodo "Data/downloads/zenodo_4628934_poultry_fecal/content (2)/imgSegmentation" \
        --v1-manifest poultryai/tasks/fecal/manifest.csv \
        --out poultryai/tasks/fecal_seg/manifest.csv

Torch-free (PIL + numpy). Reads only the JSON *headers* (not the embedded images), so it
runs over 6 800+ files in a few seconds. Run once; commit the manifest.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from poultryai.tasks.fecal import CLASS_TO_IDX
from poultryai.tasks.fecal_seg.masks import read_header


def load_v1_split(v1_manifest: Path) -> dict[str, str]:
    """basename (e.g. cocci.1819.jpg) -> split, from V1's committed manifest."""
    split_of: dict[str, str] = {}
    with open(v1_manifest, newline="") as f:
        for r in csv.DictReader(f):
            split_of[Path(r["path"]).name] = r["split"]
    return split_of


def stratified_fallback(labels: list[str], seed: int = 42) -> list[str]:
    """Stratified train/val/test for the few Zenodo images with no V1 twin."""
    rng = np.random.default_rng(seed)
    by_cls: dict[str, list[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        by_cls[lab].append(i)
    out = ["train"] * len(labels)
    for idxs in by_cls.values():
        idxs = list(idxs); rng.shuffle(idxs)
        n = len(idxs)
        n_test = max(1, round(n * 0.15)) if n > 2 else 0
        n_val = max(1, round(n * 0.15)) if n > 2 else 0
        for k, i in enumerate(idxs):
            out[i] = "test" if k < n_test else "val" if k < n_test + n_val else "train"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenodo", required=True, help="imgSegmentation dir with class subdirs")
    ap.add_argument("--v1-manifest", default="poultryai/tasks/fecal/manifest.csv")
    ap.add_argument("--out", default="poultryai/tasks/fecal_seg/manifest.csv")
    args = ap.parse_args()

    zroot = Path(args.zenodo)
    split_of = load_v1_split(Path(args.v1_manifest))
    print(f"V1 split covers {len(split_of)} filenames")

    jsons = sorted(zroot.rglob("*.json"))
    print(f"scanning {len(jsons)} Zenodo JSON headers ...")
    rows = []
    t0 = time.time(); skipped = 0
    for i, jp in enumerate(jsons):
        h = read_header(jp)
        if h is None:
            skipped += 1
            continue
        rows.append(h)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(jsons)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"parsed {len(rows)} headers ({skipped} skipped: no header/unmapped label)")

    # inherit V1 split by filename; fallback for unmatched
    unmatched = [i for i, h in enumerate(rows) if h.basename not in split_of]
    if unmatched:
        fb = stratified_fallback([rows[i].disease for i in unmatched])
        fb_map = {i: s for i, s in zip(unmatched, fb)}
    else:
        fb_map = {}
    print(f"matched to V1: {len(rows) - len(unmatched)} | unmatched (fallback): {len(unmatched)}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["json_path", "basename", "label", "label_idx", "split", "matched_v1"])
        for i, h in enumerate(rows):
            matched = h.basename in split_of
            split = split_of[h.basename] if matched else fb_map[i]
            w.writerow([h.json_path, h.basename, h.disease,
                        CLASS_TO_IDX[h.disease], split, int(matched)])

    # ---- verify + summarise ----
    per_split = defaultdict(Counter)
    for i, h in enumerate(rows):
        split = split_of.get(h.basename, fb_map.get(i))
        per_split[split][h.disease] += 1
    print("\n=== seg manifest summary (inherits V1 split) ===")
    for sp in ("train", "val", "test"):
        c = per_split[sp]; n = sum(c.values())
        dist = "  ".join(f"{k}:{c.get(k, 0)}" for k in CLASS_TO_IDX)
        print(f"  {sp:5s} n={n:5d}  ({dist})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
