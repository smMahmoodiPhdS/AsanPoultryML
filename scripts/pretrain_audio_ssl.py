#!/usr/bin/env python3
"""A2 — self-supervised pretraining of the audio encoder (masked-spectrogram autoencoder).

    python scripts/pretrain_audio_ssl.py --config configs/audio_ssl.yaml

Pretrains A1's encoder on **unlabeled** audio (the manifest's train[+val] windows — never test)
by masking log-mel patches and reconstructing them, then saves the encoder weights for A1 to
fine-tune from (``configs/audio.yaml: model.init_encoder``). Prereqs: torchaudio.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from poultryai.utils.config import load_config
from poultryai.utils.seed import set_seed
from poultryai.tasks.audio.data import make_torch_audio_dataset
from poultryai.tasks.audio.ssl import SSLConfig, batch_patch_mask, build_ssl_model, export_encoder_state


def _unlabeled_loader(d, splits, batch, workers):
    """Concatenate window datasets from the given splits (labels ignored)."""
    from torch.utils.data import ConcatDataset, DataLoader
    dss = [make_torch_audio_dataset(d["manifest"], sp, window_s=d["window_s"], hop_s=d["hop_s"],
                                    max_windows=d["max_windows"], sample_rate=d["sample_rate"],
                                    eval_mode=False) for sp in splits]
    return DataLoader(ConcatDataset(dss), batch_size=batch, shuffle=True, num_workers=workers)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/audio_ssl.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg["seed"])
    d, s, t = cfg["data"], cfg["ssl"], cfg["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 1 if args.smoke else t["max_epochs"]
    workers = 0 if args.smoke else t["num_workers"]

    loader = _unlabeled_loader(d, s["pretrain_splits"], t["batch_size"], workers)
    scfg = SSLConfig(d_model=cfg["model"]["d_model"], mask_ratio=s["mask_ratio"],
                     patch_mel=s["patch_mel"], patch_frame=s["patch_frame"],
                     sample_rate=d["sample_rate"], n_mels=cfg["model"]["n_mels"],
                     n_fft=cfg["model"]["n_fft"], hop_length=cfg["model"]["hop_length"])
    model = build_ssl_model(scfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    rng = np.random.default_rng(cfg["seed"])

    for epoch in range(epochs):
        model.train(); t0 = time.time(); run = 0.0; nb = 0
        for bi, (wav, _) in enumerate(loader):
            wav = wav.to(device)
            # derive the mel time-dim from the front-end to size the mask
            with torch.no_grad():
                mel = model.frontend(wav)
            mask = torch.from_numpy(batch_patch_mask(
                wav.shape[0], mel.shape[-2], mel.shape[-1], s["mask_ratio"],
                s["patch_mel"], s["patch_frame"], rng)).to(device)
            opt.zero_grad()
            loss, _, _ = model(wav, mask)
            loss.backward(); opt.step()
            run += loss.item(); nb += 1
            if args.smoke and bi >= 3:
                break
        sched.step()
        print(f"epoch {epoch+1}/{epochs}  ssl_recon_mse={run/max(nb,1):.4f}  ({time.time()-t0:.0f}s)")
        if args.smoke:
            break

    out = Path(cfg["export"]["encoder_path"])
    export_encoder_state(model, out)
    print(f"saved pretrained encoder -> {out}\n"
          f"set  model.init_encoder: {out}  in configs/audio.yaml to fine-tune A1 from it")


if __name__ == "__main__":
    main()
