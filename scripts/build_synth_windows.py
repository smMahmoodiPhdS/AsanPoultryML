#!/usr/bin/env python3
"""Generate the twin-synthetic multimodal dataset and report its composition (torch-free).

    python scripts/build_synth_windows.py --n-flocks 16 --flock-days 4

Windows are regenerated deterministically from ``--seed`` by ``train_fusion.py``, so this script
mainly validates the generator and writes a per-flock manifest (disease + window counts) for
provenance. Uses ``saloontwin`` for env when importable (else a built-in fallback).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from poultryai.data.schema import DISEASES, N_DISEASES
from poultryai.tasks.fusion.synth import SynthConfig, generate_dataset
from poultryai.tasks.fusion.leadtime_io import flock_key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-flocks", type=int, default=16)
    ap.add_argument("--flock-days", type=float, default=4.0)
    ap.add_argument("--healthy-frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-twin", action="store_true", help="use the built-in env fallback")
    ap.add_argument("--out", default="artifacts/fusion/synth_manifest.csv")
    args = ap.parse_args()

    cfg = SynthConfig(n_flocks=args.n_flocks, flock_days=args.flock_days,
                      healthy_frac=args.healthy_frac, seed=args.seed, use_twin=not args.no_twin)
    windows = generate_dataset(cfg)

    per_flock: dict[tuple, dict] = defaultdict(lambda: {"disease": "healthy", "n": 0, "pos": 0})
    for w in windows:
        k = flock_key(w)
        per_flock[k]["n"] += 1
        per_flock[k]["pos"] += int(w.labels.any())
        d = [i for i in range(N_DISEASES) if np.isfinite(w.onset_offset[i])]
        if d:
            per_flock[k]["disease"] = DISEASES[d[0]].value

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["owner", "saloon", "disease", "n_windows", "n_positive"])
        for (owner, saloon), v in per_flock.items():
            wr.writerow([owner, saloon, v["disease"], v["n"], v["pos"]])

    print(f"generated {len(windows)} windows from {cfg.n_flocks} flocks "
          f"({'twin' if cfg.use_twin else 'fallback'} env)")
    print("flock composition:", dict(Counter(v["disease"] for v in per_flock.values())))
    print(f"positive-labelled windows: {sum(v['pos'] for v in per_flock.values())}/{len(windows)}")
    print(f"wrote per-flock manifest -> {out}")


if __name__ == "__main__":
    main()
