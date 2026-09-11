#!/usr/bin/env python3
"""Fine-tune a YOLO bird detector on the **clip-aware** (leakage-free) Roboflow re-split.

    python scripts/train_detector.py --config configs/vision.yaml

Writes a temporary ``data.yaml`` + per-split image lists from the clip-aware manifest, then
calls Ultralytics. Requires ``pip install ultralytics``. Training on Roboflow's shipped split
would leak video frames across train/val (see build_roboflow_manifest.py); this uses the
regrouped split so reported mAP is honest.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poultryai.utils.config import load_config
from poultryai.tasks.vision.roboflow import parse_data_yaml_names


def write_split_lists(manifest: Path, work: Path) -> dict[str, Path]:
    work.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[str]] = defaultdict(list)
    with open(manifest, newline="") as f:
        for r in csv.DictReader(f):
            by_split[r["split"]].append(str(Path(r["path"]).resolve()))
    out = {}
    for split, paths in by_split.items():
        p = work / f"{split}.txt"
        p.write_text("\n".join(paths))
        out[split] = p
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vision.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dc = cfg["detector"]
    ds = dc["dataset"]

    manifest = Path(cfg["manifests_dir"]) / f"{ds}.csv"
    if not manifest.exists():
        raise SystemExit(f"missing {manifest}; run scripts/build_roboflow_manifest.py first")

    # names from the original dataset's data.yaml
    root = None
    for cand in (Path("PhdThesis/Data/downloads/Roboflow/Data/downloads") / ds,
                 Path("../Data/downloads/Roboflow/Data/downloads") / ds):
        if cand.exists():
            root = cand; break
    names = parse_data_yaml_names(root) if root else ["object"]

    work = Path(dc["out"]) / f"{ds}_clipsplit"
    lists = write_split_lists(manifest, work)
    data_yaml = work / "data.yaml"
    data_yaml.write_text(
        f"train: {lists['train'].resolve()}\n"
        f"val: {lists.get('val', lists['train']).resolve()}\n"
        f"test: {lists.get('test', lists['train']).resolve()}\n"
        f"nc: {len(names)}\n"
        f"names: {list(names)}\n")
    print(f"wrote leakage-free data.yaml -> {data_yaml}")

    from poultryai.tasks.vision.detect import train_yolo
    train_yolo(data_yaml, model=dc["model"], epochs=dc["epochs"], imgsz=dc["imgsz"],
               project=dc["out"], name=f"{ds}_detector")


if __name__ == "__main__":
    main()
