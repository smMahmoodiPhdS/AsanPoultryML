#!/usr/bin/env python3
"""Run detection → tracking → behavioural indices over a frame sequence.

Two modes:

* ``--from-labels`` — use YOLO ground-truth label files as "detections". Needs **no** detector
  or torch, so it runs now and demonstrates the tracker + index pipeline directly on the
  Roboflow data (e.g. the tracking clips).
* ``--weights <yolo.pt>`` — use a trained YOLO detector on the image frames (needs ultralytics).

Frames are grouped by source clip and ordered by frame index, chunked into windows of
``indices.window_frames``; each window yields one VISION_CHANNELS row written to a CSV that
matches ``docs/data_dictionary.md``'s ``vision.parquet`` contract.

    # runnable now (ground-truth boxes as detections):
    python scripts/extract_vision_indices.py --config configs/vision.yaml \
        --manifest poultryai/tasks/vision/manifests/roboflow_tracking.csv \
        --from-labels --out artifacts/vision/tracking_indices.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from poultryai.utils.config import load_config
from poultryai.data.schema import VISION_CHANNELS
from poultryai.tasks.vision.tracker import SortLite
from poultryai.tasks.vision.indices import Frame, window_indices


def _frame_index(path: str) -> int:
    m = re.search(r"[-_](\d+)_(?:jpg|png|jpeg)\.rf\.", Path(path).name, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", Path(path).stem)
    return int(m.group(1)) if m else 0


def _read_yolo_labels(label_path: Path, W: int, H: int):
    boxes, classes = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cid, cx, cy, w, h = int(float(p[0])), *map(float, p[1:5])
            boxes.append([(cx - w / 2) * W, (cy - h / 2) * H,
                          (cx + w / 2) * W, (cy + h / 2) * H])
            classes.append(cid)
    return np.array(boxes, dtype=float).reshape(-1, 4), np.array(classes, dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vision.yaml")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--from-labels", action="store_true")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--pose", action="store_true",
                    help="treat class ids as POSE behaviour labels (eat-drink/moving/rest). "
                         "Only set for the pose model; for a single-class bird detector the "
                         "class is NOT a behaviour, so feed_visit_rate falls back to feeder_roi.")
    ap.add_argument("--out", default="artifacts/vision/indices.csv")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ic = cfg["indices"]; tc = cfg["tracker"]
    win = int(ic["window_frames"])
    feeder = ic.get("feeder_roi")

    rows = list(csv.DictReader(open(args.manifest)))
    by_clip: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_clip[r["clip"]].append(r)

    detector = None
    if not args.from_labels:
        if not args.weights:
            raise SystemExit("provide --weights or use --from-labels")
        from poultryai.tasks.vision.detect import YoloDetector
        detector = YoloDetector(args.weights, conf=cfg["detector"]["conf"])

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    n_windows = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "window", "n_frames", *VISION_CHANNELS])
        for clip, items in by_clip.items():
            items.sort(key=lambda r: _frame_index(r["path"]))
            tracker = SortLite(iou_threshold=tc["iou_threshold"], max_age=tc["max_age"],
                               min_hits=tc["min_hits"])
            frames: list[Frame] = []
            wh = (1, 1)
            for it in items:
                with Image.open(it["path"]) as im:
                    W, H = im.size
                wh = (W, H)
                if args.from_labels:
                    boxes, classes = _read_yolo_labels(Path(it["label_path"]), W, H)
                else:
                    boxes, classes, _ = detector.detect(it["path"])
                tracks = tracker.update(boxes, classes)
                tb = np.array([t.box for t in tracks]).reshape(-1, 4)
                tids = np.array([t.track_id for t in tracks])
                behav = np.array([t.cls for t in tracks]) if args.pose else None
                frames.append(Frame(boxes=tb, track_ids=tids, behaviours=behav))
                if len(frames) >= win:
                    v = window_indices(frames, wh, feeder)
                    w.writerow([clip, n_windows, len(frames), *[f"{x:.4f}" for x in v]])
                    n_windows += 1; frames = []
            if frames:
                v = window_indices(frames, wh, feeder)
                w.writerow([clip, n_windows, len(frames), *[f"{x:.4f}" for x in v]])
                n_windows += 1
    print(f"wrote {n_windows} index windows -> {out}")


if __name__ == "__main__":
    main()
