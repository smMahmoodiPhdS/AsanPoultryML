"""A1 model: in-model log-mel front-end + CNN encoder + 3-logit head (needs torch/torchaudio).

Assembles the package's reusable audio pieces so the trained encoder can be lifted straight
into the fusion model:
    LogMelFrontend (features.audio)  →  AudioCNNEncoder (models.audio_encoder)  →  Linear(3)
The front-end is *inside* the module, so ONNX/TorchScript export takes the **raw waveform** and
no preprocessing is re-implemented on the RPi4 edge. An AST server backbone can be swapped in
via ``encoder='ast'`` once wired in ``build_audio_encoder``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import CLASSES


@dataclass(slots=True)
class AudioModelConfig:
    n_classes: int = len(CLASSES)
    d_model: int = 128
    encoder: str = "cnn"            # cnn (edge) | ast (server)
    sample_rate: int = 16_000
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 256


def build_model(cfg: AudioModelConfig, augment: bool = True):
    """Return the full waveform→logits classifier (nn.Module)."""
    import torch.nn as nn
    from poultryai.features.audio import AudioConfig, LogMelFrontend
    from poultryai.models.audio_encoder import build_audio_encoder

    acfg = AudioConfig(sample_rate=cfg.sample_rate, n_mels=cfg.n_mels, n_fft=cfg.n_fft,
                       hop_length=cfg.hop_length)
    frontend = LogMelFrontend(acfg, augment=augment)
    encoder = build_audio_encoder(cfg.encoder, cfg.d_model)

    class AudioClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.frontend = frontend
            self.encoder = encoder
            self.head = nn.Linear(cfg.d_model, cfg.n_classes)

        def embed(self, wav):
            return self.encoder(self.frontend(wav))     # (B, d_model) — the fusion-model branch

        def forward(self, wav):                          # wav: (B, n_samples)
            return self.head(self.embed(wav))

    return AudioClassifier()


def export_onnx(model, path, n_samples: int, opset: int = 17):
    """Export waveform→logits to ONNX for onnxruntime on the RPi4/VPS."""
    from pathlib import Path
    import torch
    model.eval()
    dummy = torch.randn(1, n_samples)
    path = Path(path)
    torch.onnx.export(
        model, dummy, path.as_posix(), opset_version=opset,
        input_names=["waveform"], output_names=["logits"],
        dynamic_axes={"waveform": {0: "batch", 1: "n_samples"}, "logits": {0: "batch"}})
    return path
