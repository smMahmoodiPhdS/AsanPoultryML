"""Segmentation models for V2 (needs torch; torchvision optional for DeepLab).

Two backbones, matching the thesis' edge/server split:
* **U-Net** (implemented here, no external weights) — compact, ONNX-friendly, the RPi4 edge
  model. Default.
* **DeepLabV3-ResNet50** (torchvision, pretrained) — the higher-capacity server model for
  the accuracy comparison.

Both emit per-pixel logits over ``N_SEG_CLASSES`` (background + 4 diseases).
"""
from __future__ import annotations

from dataclasses import dataclass

from . import N_SEG_CLASSES


@dataclass(slots=True)
class SegModelConfig:
    backbone: str = "unet"          # unet | deeplabv3_resnet50
    pretrained: bool = True         # only used by deeplab
    n_classes: int = N_SEG_CLASSES
    base_ch: int = 32               # unet width


def _unet(n_classes: int, base: int):
    import torch
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True))

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            b = base
            self.d1 = block(3, b); self.d2 = block(b, b * 2)
            self.d3 = block(b * 2, b * 4); self.d4 = block(b * 4, b * 8)
            self.pool = nn.MaxPool2d(2)
            self.bott = block(b * 8, b * 16)
            self.up4 = nn.ConvTranspose2d(b * 16, b * 8, 2, stride=2); self.u4 = block(b * 16, b * 8)
            self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 2, stride=2); self.u3 = block(b * 8, b * 4)
            self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 2, stride=2); self.u2 = block(b * 4, b * 2)
            self.up1 = nn.ConvTranspose2d(b * 2, b, 2, stride=2); self.u1 = block(b * 2, b)
            self.head = nn.Conv2d(b, n_classes, 1)

        def forward(self, x):
            c1 = self.d1(x); c2 = self.d2(self.pool(c1))
            c3 = self.d3(self.pool(c2)); c4 = self.d4(self.pool(c3))
            z = self.bott(self.pool(c4))
            z = self.u4(torch.cat([self.up4(z), c4], 1))
            z = self.u3(torch.cat([self.up3(z), c3], 1))
            z = self.u2(torch.cat([self.up2(z), c2], 1))
            z = self.u1(torch.cat([self.up1(z), c1], 1))
            return self.head(z)

    return UNet()


def build_seg_model(cfg: SegModelConfig):
    name = cfg.backbone.lower()
    if name == "unet":
        return _unet(cfg.n_classes, cfg.base_ch)
    if name == "deeplabv3_resnet50":
        import torch.nn as nn
        import torchvision.models.segmentation as S
        weights = S.DeepLabV3_ResNet50_Weights.DEFAULT if cfg.pretrained else None
        net = S.deeplabv3_resnet50(weights=weights)
        net.classifier[-1] = nn.Conv2d(256, cfg.n_classes, 1)
        if net.aux_classifier is not None:
            net.aux_classifier[-1] = nn.Conv2d(256, cfg.n_classes, 1)
        return _DeepLabWrap(net)
    raise ValueError(f"unknown seg backbone {cfg.backbone!r}")


class _DeepLabWrap:
    """Adapter so DeepLab (returns a dict) matches U-Net's tensor output."""
    def __new__(cls, net):
        import torch.nn as nn

        class Wrap(nn.Module):
            def __init__(self, m):
                super().__init__(); self.m = m
            def forward(self, x):
                return self.m(x)["out"]
        return Wrap(net)


def export_onnx(model, path, img_size: int = 512, opset: int = 17):
    from pathlib import Path
    import torch
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    path = Path(path)
    torch.onnx.export(
        model, dummy, path.as_posix(), opset_version=opset,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    return path
