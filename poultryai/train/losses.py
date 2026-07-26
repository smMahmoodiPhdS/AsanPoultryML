"""Losses for imbalanced multi-label disease onset.

Focal loss (Lin et al., 2017) down-weights easy negatives, which dominate because
disease-onset windows are rare. ``pos_weight`` additionally rebalances positives per
disease. Reduces to weighted BCE at gamma=0.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCEWithLogits(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None,
                 label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none",
            pos_weight=self.pos_weight if self.pos_weight is not None else None)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)          # prob of the true class
        focal = (1 - p_t).clamp_min(1e-6) ** self.gamma
        return (focal * bce).mean()
