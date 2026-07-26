"""Cross-modal fusion.

Intermediate (feature-level) fusion with a gated cross-attention block. Each modality
produces a token; a learnable query attends over the modality tokens so the model can
*down-weight a missing or unreliable modality* (e.g. camera occluded, mic saturated) —
important on a farm where any sensor can drop out. Modality dropout at train time makes
this robust; the ``present`` mask carries availability at inference.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    def __init__(self, d_model: int, n_modalities: int, heads: int = 4,
                 meta_dim: int = 0, p: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.modality_embed = nn.Parameter(torch.randn(n_modalities, d_model) * 0.02)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=p, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.meta_proj = nn.Linear(meta_dim, d_model) if meta_dim else None
        self.out_dim = d_model

    def forward(self, tokens: list[torch.Tensor],
                present: torch.Tensor | None = None,
                meta: torch.Tensor | None = None) -> torch.Tensor:
        """tokens: list of (B, d_model), one per modality (same order as embeddings).
        present: (B, n_modalities) bool mask; True = available. meta: (B, meta_dim)."""
        x = torch.stack(tokens, dim=1) + self.modality_embed        # (B,M,d)
        b = x.size(0)
        q = self.query.expand(b, -1, -1)
        key_padding = (~present.bool()) if present is not None else None
        fused, _ = self.attn(q, x, x, key_padding_mask=key_padding)  # (B,1,d)
        fused = self.norm(fused.squeeze(1))
        if self.meta_proj is not None and meta is not None:
            fused = fused + self.meta_proj(meta)
        return fused
