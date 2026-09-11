#!/usr/bin/env python3
"""Audit Roboflow's shipped splits and build a clip-aware, leakage-free re-split.

    python scripts/build_roboflow_manifest.py \
        --root PhdThesis/Data/downloads/Roboflow/Data/downloads \
        --out-dir poultryai/tasks/vision/manifests

For each dataset it (1) reports how many source clips leak across the *shipped* split, and
(2) writes a manifest with a fresh **clip-grouped** split. Torch-free; run once and commit.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poultryai.tasks.vision import DATASETS
from poultryai.tasks.vision.roboflow import (assert_clip_split_clean, audit_shipped_split,
                                             clip_aware_split, parse_data_yaml_names,
                                             scan_dataset)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing the roboflow_* datasets")
    ap.add_argument("--out-dir", default="poultryai/tasks/vision/manifests")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        ddir = root / ds
        if not ddir.exists():
            print(f"! {ds}: not found at {ddir}, skipping"); continue
        names = parse_data_yaml_names(ddir)
        imgs = scan_dataset(ddir, ds)
        audit = audit_shipped_split(imgs)
        counts = clip_aware_split(imgs, seed=args.seed)
        assert_clip_split_clean(imgs)

        out = out_dir / f"{ds}.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "label_path", "clip", "orig_split", "split",
                        "n_objects", "class_hist"])
            for im in imgs:
                w.writerow([im.path, im.label_path, im.clip, im.orig_split, im.split,
                            im.n_objects, "|".join(map(str, im.class_hist))])

        print(f"\n=== {ds} ===  classes={names}")
        print(f"  shipped split leakage: {audit['leaked_clips']}/{audit['n_clips']} "
              f"clips span >1 split  ({audit['n_images']} images)")
        print(f"  clip-aware re-split (leakage-free): {counts}")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
