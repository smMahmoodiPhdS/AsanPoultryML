#!/usr/bin/env python3
"""Train the fecal disease classifier (V1).

    python scripts/train_fecal.py --config configs/fecal.yaml

Prereqs: build the leakage-safe manifest first (scripts/build_fecal_manifest.py), then
install the training stack:  pip install -e ".[dev,track]" torchvision timm

Plain-PyTorch loop (no Lightning dependency) so it runs anywhere, including CPU for a quick
smoke test via --smoke. Logs to MLflow when ``logging.tracker: mlflow`` and mlflow is present.
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
from poultryai.tasks.fecal.data import class_weights, make_torch_dataset, read_manifest
from poultryai.tasks.fecal.eval import classification_report, format_report
from poultryai.tasks.fecal.model import FecalModelConfig, build_model, export_onnx


def _loader(manifest, split, img_size, batch, workers, train, shuffle):
    from torch.utils.data import DataLoader
    ds = make_torch_dataset(manifest, split, img_size=img_size, train=train)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=False), ds


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_p, all_y = [], []
    for x, y in loader:
        logits = model(x.to(device))
        all_p.append(torch.softmax(logits, 1).cpu().numpy())
        all_y.append(y.numpy())
    return classification_report(np.concatenate(all_p), np.concatenate(all_y))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fecal.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU run (few batches, 1 epoch) to prove the pipeline")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    d, t, m = cfg["data"], cfg["train"], cfg["model"]
    manifest = d["manifest"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 1 if args.smoke else t["max_epochs"]
    workers = 0 if args.smoke else t["num_workers"]
    out_dir = Path(cfg["logging"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    train_dl, train_ds = _loader(manifest, "train", d["img_size"], t["batch_size"],
                                 workers, train=True, shuffle=True)
    val_dl, _ = _loader(manifest, "val", d["img_size"], t["batch_size"],
                        workers, train=False, shuffle=False)

    model = build_model(FecalModelConfig(backbone=m["backbone"],
                                         pretrained=m["pretrained"],
                                         dropout=m["dropout"])).to(device)

    weight = None
    if t.get("class_weighted_loss", True):
        w = class_weights(read_manifest(manifest, "train"))
        weight = torch.tensor(w, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weight,
                                          label_smoothing=t.get("label_smoothing", 0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"],
                            weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    tracker = _maybe_mlflow(cfg, m, t)
    best_f1, best_path = -1.0, out_dir / "best.pt"
    for epoch in range(epochs):
        model.train(); t0 = time.time(); running = 0.0; nb = 0
        for bi, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward(); opt.step()
            running += loss.item(); nb += 1
            if args.smoke and bi >= 3:
                break
        sched.step()
        rep = evaluate(model, val_dl, device)
        f1 = rep.get("macro_f1", rep.get("accuracy", 0.0))
        print(f"epoch {epoch+1}/{epochs}  loss={running/max(nb,1):.4f}  "
              f"val_macro_f1={f1:.4f}  ({time.time()-t0:.0f}s)")
        if tracker:
            tracker.log_metrics({"train_loss": running/max(nb,1), "val_macro_f1": f1,
                                 "val_ece": rep["ece"]}, step=epoch)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), best_path)
        if args.smoke:
            break

    # ---- final test eval on the frozen, leakage-safe test split ----
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_dl, _ = _loader(manifest, "test", d["img_size"], t["batch_size"],
                         workers, train=False, shuffle=False)
    test_rep = evaluate(model, test_dl, device)
    print("\n=== TEST ===\n" + format_report(test_rep))
    (out_dir / "test_report.json").write_text(json.dumps(test_rep, indent=2))

    if d.get("pcr_val") and any(r.is_pcr for r in read_manifest(manifest, "test")):
        pcr_ds = make_torch_dataset(manifest, "test", img_size=d["img_size"],
                                    train=False, pcr_only=True)
        from torch.utils.data import DataLoader
        pcr_rep = evaluate(model, DataLoader(pcr_ds, batch_size=t["batch_size"]), device)
        print("\n=== TEST (PCR-confirmed slice) ===\n" + format_report(pcr_rep))
        (out_dir / "test_report_pcr.json").write_text(json.dumps(pcr_rep, indent=2))

    if cfg.get("export", {}).get("onnx") and not args.smoke:
        p = Path(cfg["export"]["onnx_path"]); p.parent.mkdir(parents=True, exist_ok=True)
        export_onnx(model, p, img_size=d["img_size"])
        print("exported ONNX ->", p)
    if tracker:
        tracker.end()
    print(f"\nbest val macro-F1 {best_f1:.4f}; checkpoint {best_path}")


def _maybe_mlflow(cfg, m, t):
    if cfg["logging"].get("tracker") != "mlflow":
        return None
    try:
        import mlflow
    except Exception:
        print("mlflow not installed; skipping tracking"); return None

    class _T:
        def __init__(self):
            mlflow.set_experiment(cfg["logging"]["experiment"])
            mlflow.start_run(run_name=cfg["run_name"])
            mlflow.log_params({**{f"model.{k}": v for k, v in m.items()},
                               **{f"train.{k}": v for k, v in t.items()}})
        def log_metrics(self, d, step): mlflow.log_metrics(d, step=step)
        def end(self): mlflow.end_run()
    return _T()


if __name__ == "__main__":
    main()
