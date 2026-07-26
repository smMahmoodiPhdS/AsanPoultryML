"""Export the trained model for the Raspberry Pi 4 edge.

Two targets:
* **ONNX** — run with onnxruntime on the RPi4 (aarch64 wheels available). Simplest path.
* **TorchScript** — if you prefer a pure-PyTorch runtime.

The log-mel front-end is part of the module, so the exported graph takes the raw
waveform (plus env/vision/meta) and emits calibrated probabilities — no Python
preprocessing needs reimplementing on the edge.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ..data.schema import ENV_CHANNELS, VISION_CHANNELS, N_DISEASES
from ..features.audio import AudioConfig
from ..models.multimodal import ModelConfig, MultimodalEarlyWarning


def _example_inputs(cfg: ModelConfig, t_env=240, t_vis=48, audio_s=4):
    sr = cfg.audio.sample_rate
    return (
        torch.randn(1, t_env, len(ENV_CHANNELS)),
        torch.randn(1, audio_s * sr),
        torch.randn(1, t_vis, len(VISION_CHANNELS)),
        torch.randn(1, cfg.meta_dim),
        torch.ones(1, 3, dtype=torch.bool),
    )


def export_onnx(model: MultimodalEarlyWarning, path: str | Path,
                cfg: ModelConfig, opset: int = 17) -> Path:
    model.eval()
    ex = _example_inputs(cfg)
    path = Path(path)
    torch.onnx.export(
        model, ex, path.as_posix(), opset_version=opset,
        input_names=["env", "audio", "vision", "meta", "present"],
        output_names=["logits"],
        dynamic_axes={"env": {1: "t_env"}, "audio": {1: "n_samples"},
                      "vision": {1: "t_vis"}},
    )
    return path


def export_torchscript(model: MultimodalEarlyWarning, path: str | Path,
                       cfg: ModelConfig) -> Path:
    model.eval()
    scripted = torch.jit.trace(model, _example_inputs(cfg), strict=False)
    path = Path(path); scripted.save(path.as_posix())
    return path


if __name__ == "__main__":
    cfg = ModelConfig()
    m = MultimodalEarlyWarning(cfg)
    out = export_onnx(m, "poultryai_edge.onnx", cfg)
    print("exported", out)
