"""Environmental time-series encoder.

Two interchangeable backbones, selectable by config:

* ``tcn``   — a dilated Temporal Convolutional Network (Bai et al., 2018). Strong,
              cheap, causal — a good fit for *streaming* edge inference on the RPi4.
* ``patch`` — a PatchTST-style patch + Transformer encoder (Nie et al., 2023). Patches
              give long effective context at low token count; channel-independent
              treatment is robust to correlated sensor channels.

Both map (B, T, C_env) -> (B, d_model) via masked mean pooling, so they are drop-in
compatible with the fusion module.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- TCN
class _Chomp(nn.Module):
    def __init__(self, k: int) -> None:
        super().__init__(); self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :-self.k] if self.k > 0 else x


class _TCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int, dilation: int, p: float) -> None:
        super().__init__()
        pad = (k - 1) * dilation
        self.net = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation)),
            _Chomp(pad), nn.GELU(), nn.Dropout(p),
            nn.utils.weight_norm(nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation)),
            _Chomp(pad), nn.GELU(), nn.Dropout(p),
        )
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.net(x) + self.down(x))


class TCNEncoder(nn.Module):
    def __init__(self, c_in: int, d_model: int = 128, levels: int = 5,
                 k: int = 3, p: float = 0.1) -> None:
        super().__init__()
        chans = [d_model] * levels
        blocks, prev = [], c_in
        for i, ch in enumerate(chans):
            blocks.append(_TCNBlock(prev, ch, k, dilation=2 ** i, p=p)); prev = ch
        self.tcn = nn.Sequential(*blocks)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # x: (B,T,C)
        h = self.tcn(x.transpose(1, 2))                       # (B,d,T)
        return h.mean(dim=-1)                                 # (B,d)


# ----------------------------------------------------------------------- PatchTST
class PatchTSTEncoder(nn.Module):
    """Channel-independent patch Transformer. Each channel is patched and encoded with
    shared weights; per-channel CLS-style means are concatenated then projected."""

    def __init__(self, c_in: int, d_model: int = 128, patch: int = 16, stride: int = 8,
                 depth: int = 3, heads: int = 8, p: float = 0.1) -> None:
        super().__init__()
        self.c_in, self.patch, self.stride = c_in, patch, stride
        self.proj = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=4 * d_model, dropout=p,
            batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(d_model * c_in, d_model)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:       # x: (B,T,C)
        b, t, c = x.shape
        x = x.transpose(1, 2)                                  # (B,C,T)
        patches = x.unfold(-1, self.patch, self.stride)        # (B,C,n_patch,patch)
        z = self.proj(patches) + self.pos                      # (B,C,n_patch,d)
        z = z.reshape(b * c, z.shape[2], -1)
        z = self.encoder(z).mean(dim=1)                        # (B*C,d)
        z = z.reshape(b, c * z.shape[-1])
        return self.head(z)                                    # (B,d)


def build_env_encoder(kind: str, c_in: int, d_model: int) -> nn.Module:
    kind = kind.lower()
    if kind == "tcn":
        return TCNEncoder(c_in, d_model)
    if kind == "patch":
        return PatchTSTEncoder(c_in, d_model)
    raise ValueError(f"unknown env encoder '{kind}' (use 'tcn' or 'patch')")
