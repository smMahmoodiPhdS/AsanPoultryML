#!/usr/bin/env python3
"""Train the acoustic health classifier (A1) — yields the reusable audio encoder.

    python scripts/build_audio_manifest.py --root <...> --out poultryai/tasks/audio/manifest.csv
    python scripts/train_audio.py --config configs/audio.yaml
    python scripts/train_audio.py --config configs/audio.yaml --smoke   # CPU pipeline check

Prereqs: pip install -e ".[dev,track,audio]" torchaudio. Plain-PyTorch loop. Reports both
window-level and recording-level metrics (the honest unit); exports waveform→logits to ONNX.
The trained ``model.encoder`` is what the multimodal fusion model reuses.
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

from poultryai.utils.config import load_config
from poultryai.utils.seed import set_seed
from poultryai.tasks.audio.data import (make_torch_audio_dataset, window_class_weights)
from poultryai.tasks.audio.eval import format_report, recording_report, window_report
from poultryai.tasks.audio.model import AudioModelConfig, build_model, export_onnx


def _loader(cfg_data, split, batch, workers, eval_mode, shuffle):
    from torch.utils.data import DataLoader
    ds = make_torch_audio_dataset(cfg_data["manifest"], split,
                                  window_s=cfg_data["window_s"], hop_s=cfg_data["hop_s"],
                                  max_windows=cfg_data["max_windows"],
                                  sample_rate=cfg_data["sample_rate"], eval_mode=eval_mode)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers), ds


@torch.no_grad()
def evaluate(model, ds, loader, device, smoke=False):
    model.eval()
    ps, ys = [], []
    for bi, (x, y) in enumerate(loader):
        ps.append(torch.softmax(model(x.to(device)), 1).cpu().numpy()); ys.append(y.numpy())
        if smoke and bi >= 2:
            break
    probs, labels = np.concatenate(ps), np.concatenate(ys)
    rec_ids = [ds.items[i].path for i in range(len(labels))]     # aligns when not shuffled
    win = window_report(probs, labels)
    rec = recording_report(probs, labels, rec_ids)
    return win, rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/audio.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg["seed"])
    d, t, m = cfg["data"], cfg["train"], cfg["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 1 if args.smoke else t["max_epochs"]
    workers = 0 if args.smoke else t["num_workers"]
    out_dir = Path(cfg["logging"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    train_dl, _ = _loader(d, "train", t["batch_size"], workers, eval_mode=False, shuffle=True)
    val_dl, val_ds = _loader(d, "val", t["batch_size"], workers, eval_mode=True, shuffle=False)

    model = build_model(AudioModelConfig(encoder=m["encoder"], d_model=m["d_model"],
                                         sample_rate=d["sample_rate"], n_mels=m["n_mels"],
                                         n_fft=m["n_fft"], hop_length=m["hop_length"]),
                        augment=True).to(device)

    # A2: optionally initialise the encoder from the self-supervised (audio-MAE) checkpoint
    init_enc = m.get("init_encoder")
    if init_enc:
        state = torch.load(init_enc, map_location=device)
        missing, unexpected = model.encoder.load_state_dict(state, strict=False)
        print(f"loaded SSL-pretrained encoder from {init_enc} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    weight = None
    if t.get("class_weighted_loss", True):
        weight = torch.tensor(window_class_weights(d["manifest"], "train", **d), device=device)
    crit = torch.nn.CrossEntropyLoss(weight=weight, label_smoothing=t.get("label_smoothing", 0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    best, best_path = -1.0, out_dir / "best.pt"
    for epoch in range(epochs):
        model.train(); t0 = time.time(); running = 0.0; nb = 0
        for bi, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
            running += loss.item(); nb += 1
            if args.smoke and bi >= 3:
                break
        sched.step()
        win, rec = evaluate(model, val_ds, val_dl, device, smoke=args.smoke)
        f1 = rec.get("macro_f1", rec.get("accuracy", 0.0))
        print(f"epoch {epoch+1}/{epochs}  loss={running/max(nb,1):.4f}  "
              f"val_rec_macro_f1={f1:.4f}  ({time.time()-t0:.0f}s)")
        if f1 > best:
            best = f1; torch.save(model.state_dict(), best_path)
        if args.smoke:
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_dl, test_ds = _loader(d, "test", t["batch_size"], workers, eval_mode=True, shuffle=False)
    win, rec = evaluate(model, test_ds, test_dl, device, smoke=args.smoke)
    print("\n=== TEST (window-level) ===\n" + format_report(win))
    print("\n=== TEST (recording-level — honest unit) ===\n" + format_report(rec))
    (out_dir / "test_window.json").write_text(json.dumps(win, indent=2))
    (out_dir / "test_recording.json").write_text(json.dumps(rec, indent=2))

    if cfg.get("export", {}).get("onnx") and not args.smoke:
        p = Path(cfg["export"]["onnx_path"]); p.parent.mkdir(parents=True, exist_ok=True)
        export_onnx(model, p, n_samples=int(d["window_s"] * d["sample_rate"]))
        print("exported ONNX ->", p)
    print(f"\nbest val recording macro-F1 {best:.4f}; checkpoint {best_path}")


if __name__ == "__main__":
    main()
