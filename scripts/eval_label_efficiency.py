#!/usr/bin/env python3
"""A2 label-efficiency study — does SSL pretraining help A1 on small labelled sets?

The scientific payoff of A2: train A1 at several **label fractions** (e.g. 10/25/50/100 %)
twice — from scratch vs. from the SSL-pretrained encoder — and plot the macro-F1 gap. SSL
should help most when labels are scarce (the low-resource-farm case).

    python scripts/pretrain_audio_ssl.py --config configs/audio_ssl.yaml     # make encoder.pt
    python scripts/eval_label_efficiency.py --config configs/audio.yaml \
        --ssl-encoder artifacts/audio_ssl/encoder.pt --fractions 0.1 0.25 0.5 1.0

Prereqs: torchaudio + scikit-learn. Uses A1's own recording-level split; the label fraction
subsamples **training recordings** (stratified), never touching val/test.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from poultryai.utils.config import load_config
from poultryai.utils.seed import set_seed
from poultryai.tasks.audio import CLASSES
from poultryai.tasks.audio.data import make_torch_audio_dataset, window_class_weights
from poultryai.tasks.audio.eval import recording_report
from poultryai.tasks.audio.model import AudioModelConfig, build_model


def _subsample_train_indices(ds, fraction: float, seed: int) -> list[int]:
    """Keep windows from a stratified ``fraction`` of *recordings* (grouped subsample)."""
    rng = np.random.default_rng(seed)
    by_cls_rec: dict[int, set[str]] = defaultdict(set)
    for it in ds.items:
        by_cls_rec[it.label_idx].add(it.path)
    keep: set[str] = set()
    for cls, recs in by_cls_rec.items():
        recs = sorted(recs); rng.shuffle(recs)
        k = max(1, int(round(len(recs) * fraction)))
        keep.update(recs[:k])
    return [i for i, it in enumerate(ds.items) if it.path in keep]


def _train_once(cfg, ssl_encoder, fraction, seed, epochs, device):
    d, t, m = cfg["data"], cfg["train"], cfg["model"]
    set_seed(seed)
    train_ds = make_torch_audio_dataset(d["manifest"], "train", window_s=d["window_s"],
                                        hop_s=d["hop_s"], max_windows=d["max_windows"],
                                        sample_rate=d["sample_rate"], eval_mode=False)
    idx = _subsample_train_indices(train_ds, fraction, seed)
    dl = DataLoader(Subset(train_ds, idx), batch_size=t["batch_size"], shuffle=True)

    model = build_model(AudioModelConfig(encoder=m["encoder"], d_model=m["d_model"],
                                         sample_rate=d["sample_rate"], n_mels=m["n_mels"],
                                         n_fft=m["n_fft"], hop_length=m["hop_length"]),
                        augment=True).to(device)
    if ssl_encoder:
        model.encoder.load_state_dict(torch.load(ssl_encoder, map_location=device), strict=False)

    w = torch.tensor(window_class_weights(d["manifest"], "train", **d), device=device)
    crit = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=t.get("label_smoothing", 0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    for _ in range(epochs):
        model.train()
        for x, y in dl:
            opt.zero_grad(); crit(model(x.to(device)), y.to(device)).backward(); opt.step()

    # recording-level test macro-F1
    test_ds = make_torch_audio_dataset(d["manifest"], "test", window_s=d["window_s"],
                                       hop_s=d["hop_s"], max_windows=d["max_windows"],
                                       sample_rate=d["sample_rate"], eval_mode=True)
    tdl = DataLoader(test_ds, batch_size=t["batch_size"])
    model.eval(); ps, ys, ids = [], [], []
    with torch.no_grad():
        off = 0
        for x, y in tdl:
            ps.append(torch.softmax(model(x.to(device)), 1).cpu().numpy()); ys.append(y.numpy())
            ids += [test_ds.items[off + j].path for j in range(len(y))]; off += len(y)
    rep = recording_report(np.concatenate(ps), np.concatenate(ys), ids)
    return rep.get("macro_f1", rep.get("accuracy", 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/audio.yaml")
    ap.add_argument("--ssl-encoder", required=True)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default="artifacts/audio/label_efficiency.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    print(f"{'frac':>6} {'scratch_f1':>11} {'ssl_f1':>9} {'delta':>8}")
    for frac in args.fractions:
        sc = np.mean([_train_once(cfg, None, frac, s, args.epochs, device) for s in args.seeds])
        ss = np.mean([_train_once(cfg, args.ssl_encoder, frac, s, args.epochs, device)
                      for s in args.seeds])
        rows.append({"fraction": frac, "scratch_macro_f1": float(sc),
                     "ssl_macro_f1": float(ss), "delta": float(ss - sc)})
        print(f"{frac:6.2f} {sc:11.4f} {ss:9.4f} {ss-sc:+8.4f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}  (SSL should help most at low label fractions)")


if __name__ == "__main__":
    main()
