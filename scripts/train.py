#!/usr/bin/env python3
"""Training entry point.

    python scripts/train.py --config configs/default.yaml

Uses PyTorch Lightning if available (recommended) for logging, checkpointing, mixed
precision, and early stopping. Data loading is delegated to ``poultryai.data``; plug in
your dataset builder at ``load_windows`` once real data is collected.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from poultryai.utils.config import load_config
from poultryai.utils.seed import set_seed
from poultryai.data.datamodule import WindowDataset, split_by_flock, collate
from poultryai.models.multimodal import ModelConfig, LitEarlyWarning
from poultryai.features.audio import AudioConfig


def load_windows(root: str):
    """TODO: replace with the real loader once data is collected.

    Must return a list[poultryai.data.schema.Window]. Until then this raises to avoid
    silently training on nothing."""
    raise NotImplementedError(
        f"Implement load_windows('{root}') to yield schema.Window objects "
        "(see docs/data_dictionary.md for the on-disk contract).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    windows = load_windows(cfg["data"]["root"])
    train_w, val_w, test_w = split_by_flock(
        windows, cfg["data"]["val_frac"], cfg["data"]["test_frac"], cfg["seed"])

    train_ds = WindowDataset(train_w)
    mean, std = train_ds.fit_normalisation()
    val_ds = WindowDataset(val_w, mean, std)

    import torch
    from torch.utils.data import DataLoader
    dl_kw = dict(batch_size=cfg["train"]["batch_size"], collate_fn=collate, num_workers=4)
    train_dl = DataLoader(train_ds, shuffle=True, **dl_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_kw)

    mc = cfg["model"]
    model_cfg = ModelConfig(
        d_model=mc["d_model"], env_backbone=mc["env_backbone"],
        audio_backbone=mc["audio_backbone"], meta_dim=mc["meta_dim"],
        heads=mc["heads"], dropout=mc["dropout"],
        modality_dropout=mc["modality_dropout"],
        audio=AudioConfig(sample_rate=mc["audio"]["sample_rate"],
                          n_mels=mc["audio"]["n_mels"], n_fft=mc["audio"]["n_fft"],
                          hop_length=mc["audio"]["hop_length"]))
    lit = LitEarlyWarning(model_cfg, lr=cfg["train"]["lr"],
                          weight_decay=cfg["train"]["weight_decay"],
                          focal_gamma=cfg["train"]["focal_gamma"])

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    ckpt = ModelCheckpoint(monitor=cfg["train"]["early_stop_metric"],
                           mode=cfg["train"]["early_stop_mode"], save_top_k=1)
    es = EarlyStopping(monitor=cfg["train"]["early_stop_metric"],
                       mode=cfg["train"]["early_stop_mode"], patience=8)
    trainer = pl.Trainer(max_epochs=cfg["train"]["max_epochs"],
                         precision=cfg["train"]["precision"], callbacks=[ckpt, es],
                         log_every_n_steps=10)
    trainer.fit(lit, train_dl, val_dl)
    print("best checkpoint:", ckpt.best_model_path)


if __name__ == "__main__":
    main()
