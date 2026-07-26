"""Transfer-learning backbone for the fecal classifier (needs torch; timm optional).

A pretrained CNN with a fresh linear head over the 4 fecal classes. EfficientNet-B0 and
ResNet-18 are the recommended edge-friendly defaults (both export cleanly to ONNX for the
RPi4); a ConvNeXt/MobileViT can be swapped in via ``timm`` for the accuracy/size ablation
the plan calls for. The exported graph takes a normalised image tensor and emits 4 logits.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import CLASSES


@dataclass(slots=True)
class FecalModelConfig:
    backbone: str = "efficientnet_b0"   # efficientnet_b0 | resnet18 | any timm name
    pretrained: bool = True
    n_classes: int = len(CLASSES)
    dropout: float = 0.2


def build_model(cfg: FecalModelConfig):
    """Return an ``nn.Module``. Uses torchvision for the two defaults, timm otherwise."""
    import torch.nn as nn

    name = cfg.backbone.lower()
    if name in ("efficientnet_b0", "resnet18"):
        import torchvision.models as M
        if name == "efficientnet_b0":
            weights = M.EfficientNet_B0_Weights.DEFAULT if cfg.pretrained else None
            net = M.efficientnet_b0(weights=weights)
            in_f = net.classifier[1].in_features
            net.classifier = nn.Sequential(nn.Dropout(cfg.dropout),
                                           nn.Linear(in_f, cfg.n_classes))
        else:
            weights = M.ResNet18_Weights.DEFAULT if cfg.pretrained else None
            net = M.resnet18(weights=weights)
            net.fc = nn.Sequential(nn.Dropout(cfg.dropout),
                                   nn.Linear(net.fc.in_features, cfg.n_classes))
        return net

    # fall back to timm for anything else (convnext_tiny, mobilevit_s, ...)
    import timm
    return timm.create_model(name, pretrained=cfg.pretrained,
                             num_classes=cfg.n_classes, drop_rate=cfg.dropout)


def export_onnx(model, path, img_size: int = 224, opset: int = 17):
    """Export to ONNX for onnxruntime on the RPi4/VPS (mirrors inference/export_edge.py)."""
    from pathlib import Path
    import torch
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    path = Path(path)
    torch.onnx.export(
        model, dummy, path.as_posix(), opset_version=opset,
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    return path
