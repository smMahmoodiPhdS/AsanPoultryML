#!/usr/bin/env python3
"""Build the recording-level manifest for the audio classifier (A1).

    python scripts/build_audio_manifest.py \
        --root PhdThesis/Data/downloads/mendeley_vocalization/Chicken_Audio_Dataset \
        --out  poultryai/tasks/audio/manifest.csv

Torch-free (uses the ``wave`` stdlib). Splits at the recording level (stratified by class) and
records each clip's estimated window count under the config's windowing params, so the split's
effective (windowed) class balance is visible. Run once; commit the manifest.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poultryai.tasks.audio import CLASS_TO_IDX
from poultryai.tasks.audio.splits import (assert_recording_level, scan_corpus,
                                          stratified_split)
from poultryai.tasks.audio.windowing import n_windows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="poultryai/tasks/audio/manifest.csv")
    ap.add_argument("--window-s", type=float, default=4.0)
    ap.add_argument("--hop-s", type=float, default=2.0, help="train-time window hop (overlap)")
    ap.add_argument("--max-windows", type=int, default=20, help="cap windows per recording")
    ap.add_argument("--sample-rate", type=int, default=16000, help="target sr for windowing")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    recs = scan_corpus(args.root)
    if not recs:
        raise SystemExit(f"no wavs under {args.root}")
    counts = stratified_split(recs, args.val_frac, args.test_frac, args.seed)
    assert_recording_level(recs)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "label_idx", "duration_s", "sample_rate",
                    "n_windows_train", "split"])
        for r in recs:
            nw = n_windows(r.duration_s, args.window_s, args.hop_s, args.max_windows,
                           args.sample_rate)
            w.writerow([r.path, r.label, CLASS_TO_IDX[r.label], f"{r.duration_s:.2f}",
                        r.sample_rate, nw, r.split])

    print(f"recordings: {len(recs)}  split(files): {counts}")
    print("\n=== per-split (recordings | est. train windows) ===")
    for sp in ("train", "val", "test"):
        rs = [r for r in recs if r.split == sp]
        by = Counter(r.label for r in rs)
        wins = Counter()
        for r in rs:
            wins[r.label] += n_windows(r.duration_s, args.window_s, args.hop_s,
                                       args.max_windows, args.sample_rate)
        files = "  ".join(f"{c}:{by.get(c, 0)}" for c in CLASS_TO_IDX)
        winstr = "  ".join(f"{c}:{wins.get(c, 0)}" for c in CLASS_TO_IDX)
        print(f"  {sp:5s} files=({files})  windows=({winstr})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
