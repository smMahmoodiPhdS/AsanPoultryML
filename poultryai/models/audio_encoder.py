"""Audio encoder over log-mel spectrograms.

A compact convolutional encoder (ResNet-style stem + depthwise-separable blocks) that
runs on the RPi4 edge. It consumes the (B,1,n_mels,n_frames) tensor from
``features.audio.LogMelFrontend`` and returns a (B, d_model) embedding.

We deliberately keep this CNN small and separable so the *whole* model exports to
TFLite/ONNX and fits the edge. A full Audio Spectrogram Transformer can be swapped in
(``kind='ast'``) for the server-side model where compute is not constrained.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _SepConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, 3, stride=stride, padding=1, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class AudioCNNEncoder(nn.Module):
    def __init__(self, d_model: int = 128, widths: tuple[int, ...] = (32, 64, 96, 128)) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, widths[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]), nn.GELU())
        blocks, prev = [], widths[0]
        for w in widths:
            blocks += [_SepConv(prev, w, stride=2), _SepConv(w, w)]
            prev = w
        self.body = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(prev, d_model)
        self.out_dim = d_model

    def forward(self, mel: torch.Tensor) -> torch.Tensor:     # (B,1,M,F)
        h = self.body(self.stem(mel))
        h = self.pool(h).flatten(1)
        return self.head(h)


def build_audio_encoder(kind: str = "cnn", d_model: int = 128) -> nn.Module:
    if kind == "cnn":
        return AudioCNNEncoder(d_model)
    if kind == "ast":
        # Optional heavy backbone for the server model; import lazily to keep edge light.
        raise NotImplementedError(
            "Wire a torchaudio/HF Audio Spectrogram Transformer here for the server model.")
    raise ValueError(f"unknown audio encoder '{kind}'")
