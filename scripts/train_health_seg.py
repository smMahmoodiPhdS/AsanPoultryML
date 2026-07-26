#!/usr/bin/env python3
"""Train the healthy/sick segmentation model (V4).

    python scripts/train_health_seg.py --config configs/health_seg.yaml
    python scripts/train_health_seg.py --config configs/health_seg.yaml --smoke

Reuses V2's U-Net/DeepLab backbones (``poultryai.tasks.fecal_seg.model``) with 3 classes and
V3's clip-aware healthy_sick split. Loss = pixel-weighted CE + soft Dice; selects on
foreground mIoU; reports segmentation metrics + a derived image-level health report.
Prereqs: pip install -e ".[dev,track]" torchvision.
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
import torch.nn.functional as F

from poultryai.utils.config import load_config
from poultryai.utils.seed import set_seed
from poultryai.tasks.health_seg import N_SEG_CLASSES
from poultryai.tasks.health_seg.data import (make_torch_seg_dataset, read_manifest,
                                             seg_class_pixel_weights)
from poultryai.tasks.health_seg.eval import (dominant_health, format_seg_metrics,
                                             image_health_report, seg_metrics,
                                             update_confusion)
from poultryai.tasks.fecal_seg.model import SegModelConfig, build_seg_model, export_onnx


def soft_dice_loss(logits, target, eps: float = 1e-6):
    prob = F.softmax(logits, dim=1)
    onehot = F.one_hot(target, N_SEG_CLASSES).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (prob * onehot).sum(dims)
    denom = prob.sum(dims) + onehot.sum(dims)
    return (1 - (2 * inter + eps) / (denom + eps)).mean()


def _loader(manifest, split, img_size, batch, workers, train, shuffle):
    from torch.utils.data import DataLoader
    ds = make_torch_seg_dataset(manifest, split, img_size=img_size, train=train)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers)


@torch.no_grad()
def evaluate(model, loader, device, smoke=False):
    model.eval()
    conf = np.zeros((N_SEG_CLASSES, N_SEG_CLASSES), dtype=np.int64)
    ip, it = [], []
    for bi, (x, y) in enumerate(loader):
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        yt = y.numpy()
        update_confusion(conf, pred, yt)
        for p, t in zip(pred, yt):
            ip.append(dominant_health(p)); it.append(dominant_health(t))
        if smoke and bi >= 2:
            break
    m = seg_metrics(conf)
    m["image_level"] = image_health_report(np.array(ip), np.array(it))
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/health_seg.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg["seed"])
    d, t, m = cfg["data"], cfg["train"], cfg["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 1 if args.smoke else t["max_epochs"]
    workers = 0 if args.smoke else t["num_workers"]
    out_dir = Path(cfg["logging"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    train_dl = _loader(d["manifest"], "train", d["img_size"], t["batch_size"], workers, True, True)
    val_dl = _loader(d["manifest"], "val", d["img_size"], t["batch_size"], workers, False, False)

    model = build_seg_model(SegModelConfig(backbone=m["backbone"], pretrained=m["pretrained"],
                                           n_classes=N_SEG_CLASSES,
                                           base_ch=m.get("base_ch", 32))).to(device)

    weight = None
    if t.get("class_weighted_loss", True):
        w = seg_class_pixel_weights(d["manifest"], "train", img_size=d["img_size"],
                                    sample=30 if args.smoke else 200)
        weight = torch.tensor(w, device=device)
        print("pixel class weights (bg/healthy/sick):", [round(float(x), 3) for x in w])
    ce = torch.nn.CrossEntropyLoss(weight=weight)
    dice_w = t.get("dice_weight", 0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    best, best_path = -1.0, out_dir / "best.pt"
    for epoch in range(epochs):
        model.train(); t0 = time.time(); running = 0.0; nb = 0
        for bi, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = ce(logits, y) + dice_w * soft_dice_loss(logits, y)
            loss.backward(); opt.step()
            running += loss.item(); nb += 1
            if args.smoke and bi >= 2:
                break
        sched.step()
        vm = evaluate(model, val_dl, device, smoke=args.smoke)
        sel = vm["mIoU_foreground"]
        print(f"epoch {epoch+1}/{epochs}  loss={running/max(nb,1):.4f}  "
              f"val_mIoU_fg={sel:.4f}  img_acc={vm['image_level']['accuracy']:.3f}  "
              f"({time.time()-t0:.0f}s)")
        if sel > best:
            best = sel; torch.save(model.state_dict(), best_path)
        if args.smoke:
            break

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_dl = _loader(d["manifest"], "test", d["img_size"], t["batch_size"], workers, False, False)
    tm = evaluate(model, test_dl, device, smoke=args.smoke)
    print("\n=== TEST (segmentation) ===\n" + format_seg_metrics(tm))
    il = tm["image_level"]
    print(f"\n=== TEST (image-level health) ===\naccuracy={il['accuracy']:.4f}  recall={il['recall']}")
    (out_dir / "test_report.json").write_text(json.dumps(tm, indent=2))

    if cfg.get("export", {}).get("onnx") and not args.smoke:
        p = Path(cfg["export"]["onnx_path"]); p.parent.mkdir(parents=True, exist_ok=True)
        export_onnx(model, p, img_size=d["img_size"]); print("exported ONNX ->", p)
    print(f"\nbest val foreground-mIoU {best:.4f}; checkpoint {best_path}")


if __name__ == "__main__":
    main()
