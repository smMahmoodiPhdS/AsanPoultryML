"""Full multimodal early-warning model + LightningModule.

Pipeline:  env -> TCN/Patch ─┐
           audio -> LogMel+CNN ┼─ AttentionFusion(+meta) -> per-disease heads
           vision -> MLP ──────┘

Output: per-disease logits (multi-label). Trained with focal loss to handle the heavy
class imbalance of rare disease-onset windows, plus temperature-scaling calibration
applied post-hoc (see ``train.metrics`` / ``eval.calibration``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from ..data.schema import ENV_CHANNELS, VISION_CHANNELS, N_DISEASES
from ..features.audio import AudioConfig, LogMelFrontend
from .audio_encoder import build_audio_encoder
from .env_encoder import build_env_encoder
from .fusion import AttentionFusion


@dataclass
class ModelConfig:
    d_model: int = 128
    env_backbone: str = "patch"        # 'tcn' | 'patch'
    audio_backbone: str = "cnn"        # 'cnn' | 'ast'
    meta_dim: int = 3
    heads: int = 4
    dropout: float = 0.1
    modality_dropout: float = 0.2      # train-time random modality masking
    audio: AudioConfig = field(default_factory=AudioConfig)


class MultimodalEarlyWarning(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.env_enc = build_env_encoder(cfg.env_backbone, len(ENV_CHANNELS), d)
        self.logmel = LogMelFrontend(cfg.audio, augment=True)
        self.audio_enc = build_audio_encoder(cfg.audio_backbone, d)
        self.vision_enc = nn.Sequential(
            nn.Linear(len(VISION_CHANNELS), d), nn.GELU(), nn.LayerNorm(d))
        self.fusion = AttentionFusion(d, n_modalities=3, heads=cfg.heads,
                                      meta_dim=cfg.meta_dim, p=cfg.dropout)
        # One linear head per disease keeps per-disease calibration independent.
        self.heads = nn.ModuleList([nn.Linear(d, 1) for _ in range(N_DISEASES)])

    def _modality_dropout(self, present: torch.Tensor) -> torch.Tensor:
        if not self.training or self.cfg.modality_dropout <= 0:
            return present
        drop = torch.rand_like(present.float()) < self.cfg.modality_dropout
        keep = present & ~drop
        # never drop *all* modalities for a sample
        empty = keep.sum(dim=1) == 0
        keep[empty] = present[empty]
        return keep

    def forward(self, env: torch.Tensor, audio: torch.Tensor, vision: torch.Tensor,
                meta: torch.Tensor, present: torch.Tensor | None = None) -> torch.Tensor:
        b = env.size(0)
        if present is None:
            present = torch.ones(b, 3, dtype=torch.bool, device=env.device)
        present = self._modality_dropout(present)
        tok_env = self.env_enc(env)
        tok_aud = self.audio_enc(self.logmel(audio))
        tok_vis = self.vision_enc(vision.mean(dim=1))     # pool vision timesteps
        fused = self.fusion([tok_env, tok_aud, tok_vis], present=present, meta=meta)
        logits = torch.cat([h(fused) for h in self.heads], dim=1)   # (B, N_DISEASES)
        return logits


# ---------------------------------------------------------------- LightningModule
try:
    import pytorch_lightning as pl  # type: ignore
    _BASE = pl.LightningModule
except Exception:                    # keep importable without Lightning installed
    _BASE = nn.Module


class LitEarlyWarning(_BASE):  # type: ignore[misc]
    def __init__(self, model_cfg: ModelConfig, lr: float = 3e-4, weight_decay: float = 1e-2,
                 focal_gamma: float = 2.0, pos_weight: list[float] | None = None) -> None:
        super().__init__()
        from ..train.losses import FocalBCEWithLogits
        from ..train.metrics import MultiLabelMetrics
        self.save_hyperparameters(ignore=["model_cfg"]) if hasattr(self, "save_hyperparameters") else None
        self.model = MultimodalEarlyWarning(model_cfg)
        pw = torch.tensor(pos_weight) if pos_weight else None
        self.loss = FocalBCEWithLogits(gamma=focal_gamma, pos_weight=pw)
        self.metrics = MultiLabelMetrics(N_DISEASES)
        self.lr, self.wd = lr, weight_decay

    def forward(self, batch):  # convenience
        return self.model(batch["env"], batch["audio"], batch["vision"],
                          batch["meta"], batch.get("present"))

    def _step(self, batch, stage: str):
        logits = self(batch)
        loss = self.loss(logits, batch["labels"].float())
        if hasattr(self, "log"):
            self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)
        self.metrics.update(logits.detach(), batch["labels"])
        return loss

    def training_step(self, batch, _):   return self._step(batch, "train")
    def validation_step(self, batch, _): return self._step(batch, "val")

    def on_validation_epoch_end(self):
        res = self.metrics.compute(); self.metrics.reset()
        if hasattr(self, "log_dict"):
            self.log_dict({f"val/{k}": v for k, v in res.items()})

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        return {"optimizer": opt, "lr_scheduler": sch}
