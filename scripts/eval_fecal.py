#!/usr/bin/env python3
"""Evaluate a trained fecal checkpoint on the frozen test split (and PCR slice).

    python scripts/eval_fecal.py --config configs/fecal.yaml --ckpt artifacts/fecal/best.pt

Separate from training so results can be regenerated from a checkpoint for the thesis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from poultryai.utils.config import load_config
from poultryai.utils.seed import set_seed
from poultryai.tasks.fecal.data import make_torch_dataset, read_manifest
from poultryai.tasks.fecal.eval import classification_report, format_report
from poultryai.tasks.fecal.model import FecalModelConfig, build_model


@torch.no_grad()
def _infer(model, loader, device):
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        ps.append(torch.softmax(model(x.to(device)), 1).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ps), np.concatenate(ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fecal.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg["seed"])
    d, m = cfg["data"], cfg["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(FecalModelConfig(backbone=m["backbone"], pretrained=False,
                                         dropout=m["dropout"])).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    ds = make_torch_dataset(d["manifest"], args.split, img_size=d["img_size"], train=False)
    probs, labels = _infer(model, DataLoader(ds, batch_size=cfg["train"]["batch_size"]), device)
    rep = classification_report(probs, labels)
    print(f"=== {args.split.upper()} ===\n" + format_report(rep))

    out = Path(cfg["logging"]["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.split}_report.json").write_text(json.dumps(rep, indent=2))
    if d.get("pcr_val") and any(r.is_pcr for r in read_manifest(d["manifest"], args.split)):
        pds = make_torch_dataset(d["manifest"], args.split, img_size=d["img_size"],
                                 train=False, pcr_only=True)
        p2, y2 = _infer(model, DataLoader(pds, batch_size=cfg["train"]["batch_size"]), device)
        rep2 = classification_report(p2, y2)
        print("\n=== PCR-confirmed slice ===\n" + format_report(rep2))
        (out / f"{args.split}_report_pcr.json").write_text(json.dumps(rep2, indent=2))


if __name__ == "__main__":
    main()
