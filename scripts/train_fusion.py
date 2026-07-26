#!/usr/bin/env python3
"""C1 — train the multimodal onset model on twin-synthetic data + lead-time evaluation.

    python scripts/train_fusion.py --config configs/fusion.yaml
    python scripts/train_fusion.py --config configs/fusion.yaml --smoke
    python scripts/train_fusion.py --config configs/fusion.yaml --ablation audio_off

Regenerates the synthetic dataset (deterministic by seed) via ``tasks.fusion.synth``, splits by
flock (no leakage), trains ``MultimodalEarlyWarning`` with focal loss (plain-torch loop, no
Lightning dependency), then runs the **lead-time protocol** (`eval.leadtime`): tune the alarm
threshold on val flocks to a false-alarm budget, freeze it, and report per-disease median lead
time on test. ``--ablation {audio_off,vision_off,env_off}`` zeroes a modality's presence mask to
quantify its marginal value. Prereqs: torch + torchaudio.

Framing: **simulation** to validate the C1 pipeline; absolute lead times are meaningful only on
real / twin-calibrated data.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from poultryai.utils.config import load_config
from poultryai.utils.seed import set_seed
from poultryai.data.datamodule import WindowDataset, collate, split_by_flock
from poultryai.data.schema import DISEASES, N_DISEASES
from poultryai.features.audio import AudioConfig
from poultryai.models.multimodal import ModelConfig, MultimodalEarlyWarning
from poultryai.train.losses import FocalBCEWithLogits
from poultryai.eval.leadtime import evaluate_leadtime, tune_threshold_for_fpr
from poultryai.tasks.fusion.synth import SynthConfig, generate_dataset
from poultryai.tasks.fusion.leadtime_io import build_flock_series

_ABLATE = {"env_off": 0, "audio_off": 1, "vision_off": 2}


def _score_windows(model, windows, ablation, device, batch=16):
    """Per-window sigmoid scores (n_windows, N_DISEASES), windows kept in given order."""
    ds = WindowDataset(windows, model._env_mean, model._env_std)
    dl = DataLoader(ds, batch_size=batch, collate_fn=collate, shuffle=False)
    model.eval(); out = []
    with torch.no_grad():
        for b in dl:
            present = b["present"].clone()
            if ablation in _ABLATE:
                present[:, _ABLATE[ablation]] = False
            logits = model(b["env"].to(device), b["audio"].to(device), b["vision"].to(device),
                           b["meta"].to(device), present.to(device))
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


def leadtime_report(model, windows, ablation, device, target_fpr, persistence,
                    tune_windows=None):
    """Tune threshold on ``tune_windows`` (val) if given, else on ``windows``; eval on windows."""
    scores = _score_windows(model, windows, ablation, device)
    tune_scores = (_score_windows(model, tune_windows, ablation, device)
                   if tune_windows is not None else scores)
    rows = {}
    for d in range(N_DISEASES):
        fs_test = build_flock_series(windows, scores, d)
        if not any(f.is_diseased for f in fs_test):
            continue
        fs_tune = build_flock_series(tune_windows or windows, tune_scores, d)
        thr = tune_threshold_for_fpr(fs_tune, target_fpr, persistence)
        r = evaluate_leadtime(fs_test, thr, persistence)
        rows[DISEASES[d].value] = {"threshold": thr, "median_lead_h": r.median_lead_s / 3600,
                                   "detected_before_onset": r.detected_before_onset,
                                   "flock_fpr": r.flock_false_positive_rate,
                                   "n_diseased": r.n_diseased, "n_healthy": r.n_healthy}
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fusion.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ablation", choices=list(_ABLATE) + ["none"], default="none")
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    s = cfg["synth"]
    scfg = SynthConfig(n_flocks=8 if args.smoke else s["n_flocks"],
                       healthy_frac=s["healthy_frac"], flock_days=2.0 if args.smoke else s["flock_days"],
                       window_minutes=s["window_minutes"], stride_minutes=s["stride_minutes"],
                       label_horizon_h=s["label_horizon_h"], pre_onset_horizon_h=s["pre_onset_horizon_h"],
                       sample_rate=cfg["model"]["audio"]["sample_rate"], seed=cfg["seed"])
    print("generating twin-synthetic dataset ...")
    windows = generate_dataset(scfg)
    train_w, val_w, test_w = split_by_flock(windows, cfg["data"]["val_frac"],
                                             cfg["data"]["test_frac"], cfg["seed"])
    print(f"windows: {len(windows)} | train {len(train_w)} val {len(val_w)} test {len(test_w)}")

    train_ds = WindowDataset(train_w)
    mean, std = train_ds.fit_normalisation()
    dl = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], collate_fn=collate,
                    shuffle=True)

    mc = cfg["model"]
    model_cfg = ModelConfig(d_model=mc["d_model"], env_backbone=mc["env_backbone"],
                            audio_backbone=mc["audio_backbone"], meta_dim=mc["meta_dim"],
                            heads=mc["heads"], dropout=mc["dropout"],
                            modality_dropout=mc["modality_dropout"],
                            audio=AudioConfig(sample_rate=mc["audio"]["sample_rate"],
                                              n_mels=mc["audio"]["n_mels"], n_fft=mc["audio"]["n_fft"],
                                              hop_length=mc["audio"]["hop_length"]))
    model = MultimodalEarlyWarning(model_cfg).to(device)
    model._env_mean, model._env_std = mean, std          # stash for scoring
    crit = FocalBCEWithLogits(gamma=cfg["train"]["focal_gamma"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    epochs = 1 if args.smoke else cfg["train"]["max_epochs"]

    for epoch in range(epochs):
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        for bi, b in enumerate(dl):
            opt.zero_grad()
            logits = model(b["env"].to(device), b["audio"].to(device), b["vision"].to(device),
                           b["meta"].to(device), b["present"].to(device))
            loss = crit(logits, b["labels"].to(device))
            loss.backward(); opt.step(); run += loss.item(); nb += 1
            if args.smoke and bi >= 3:
                break
        print(f"epoch {epoch+1}/{epochs}  focal_loss={run/max(nb,1):.4f}  ({time.time()-t0:.0f}s)")
        if args.smoke:
            break

    # ---- lead-time protocol: tune on val, evaluate on test ----
    tfpr = cfg["eval"]["target_flock_fpr"]; pers = cfg["eval"]["persistence"]
    print("\n=== lead-time (test; threshold tuned on val @ FPR "
          f"{tfpr:.0%}) — ABLATION={args.ablation} ===")
    rep = leadtime_report(model, test_w, args.ablation, device, tfpr, pers, tune_windows=val_w)
    for name, r in rep.items():
        print(f"  {name:22s} thr={r['threshold']:.2f} med_lead={r['median_lead_h']:5.1f}h "
              f"detect%={r['detected_before_onset']:.0%} FPR={r['flock_fpr']:.0%}")
    out = Path(cfg["logging"]["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    (out / f"leadtime_{args.ablation}.json").write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {out}/leadtime_{args.ablation}.json")
    print("NB: simulation — absolute lead times are pipeline validation, not a scientific result.")


if __name__ == "__main__":
    main()
