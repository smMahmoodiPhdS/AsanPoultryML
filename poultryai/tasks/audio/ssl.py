"""A2 — Self-supervised audio pretraining (masked-spectrogram autoencoder / audio-MAE).

The highest-leverage addition for a small labelled set: pretrain **A1's exact encoder**
(`AudioCNNEncoder` stem+body) on *unlabeled* audio by masking patches of the log-mel
spectrogram and reconstructing them, then fine-tune A1 from those weights. This is the
label-efficiency contribution in `docs/ai-model-plan.md` (A2).

Design
------
* Front-end and encoder are the **same** modules A1 uses (`features.audio.LogMelFrontend`,
  `models.audio_encoder.AudioCNNEncoder`), so the pretrained conv weights transfer 1:1 into A1.
* SSL task: zero out random time-frequency **patches** of the log-mel (~60 %), encode the
  corrupted spectrogram, decode back to the full spectrogram, and take the reconstruction MSE
  **on the masked region** (the MAE signal). No labels are used.
* The patch-mask generator is numpy-only (torch-free) so it is unit-testable.
* ``export_encoder_state`` writes a checkpoint loadable into A1's ``model.encoder``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------------------
# Masking (torch-free — unit-tested)
# --------------------------------------------------------------------------------------
def random_patch_mask(n_mels: int, n_frames: int, patch_mel: int = 8, patch_frame: int = 8,
                      mask_ratio: float = 0.6, rng: np.random.Generator | None = None
                      ) -> np.ndarray:
    """Boolean (n_mels, n_frames) mask; True = masked. ~``mask_ratio`` of patches masked.

    Masks whole time-frequency *patches* (not pixels) so the encoder must use context to
    inpaint, as in MAE. Partial edge patches are handled by clipping.
    """
    rng = rng or np.random.default_rng()
    n_pm = int(np.ceil(n_mels / patch_mel))
    n_pf = int(np.ceil(n_frames / patch_frame))
    n_patches = n_pm * n_pf
    n_mask = int(round(mask_ratio * n_patches))
    masked = rng.choice(n_patches, size=n_mask, replace=False)
    mask = np.zeros((n_mels, n_frames), dtype=bool)
    for p in masked:
        i, j = divmod(int(p), n_pf)
        mask[i * patch_mel:(i + 1) * patch_mel, j * patch_frame:(j + 1) * patch_frame] = True
    return mask


def batch_patch_mask(batch: int, n_mels: int, n_frames: int, mask_ratio: float = 0.6,
                     patch_mel: int = 8, patch_frame: int = 8,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """(B, 1, n_mels, n_frames) float mask (1.0 = masked), one independent mask per item."""
    rng = rng or np.random.default_rng()
    out = np.zeros((batch, 1, n_mels, n_frames), dtype=np.float32)
    for b in range(batch):
        out[b, 0] = random_patch_mask(n_mels, n_frames, patch_mel, patch_frame,
                                      mask_ratio, rng).astype(np.float32)
    return out


# --------------------------------------------------------------------------------------
# Model (torch — imported lazily)
# --------------------------------------------------------------------------------------
@dataclass(slots=True)
class SSLConfig:
    d_model: int = 128
    mask_ratio: float = 0.6
    patch_mel: int = 8
    patch_frame: int = 8
    sample_rate: int = 16_000
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 256
    decoder_ch: int = 64


def build_ssl_model(cfg: SSLConfig = SSLConfig()):
    """Masked-spectrogram autoencoder. ``.encoder`` is an ``AudioCNNEncoder`` (transfers to A1)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from poultryai.features.audio import AudioConfig, LogMelFrontend
    from poultryai.models.audio_encoder import AudioCNNEncoder

    acfg = AudioConfig(sample_rate=cfg.sample_rate, n_mels=cfg.n_mels, n_fft=cfg.n_fft,
                       hop_length=cfg.hop_length)

    class MaskedSpecAutoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            # front-end WITHOUT SpecAugment (SSL masking is our augmentation)
            self.frontend = LogMelFrontend(acfg, augment=False)
            self.encoder = AudioCNNEncoder(cfg.d_model)          # the transferable encoder
            c = cfg.decoder_ch
            # decoder: upsample the pre-pool feature map back toward the spectrogram
            feat_ch = 128                                        # AudioCNNEncoder last width
            self.decoder = nn.Sequential(
                nn.Conv2d(feat_ch, c * 4, 3, padding=1), nn.GELU(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(c * 4, c * 2, 3, padding=1), nn.GELU(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(c * 2, c, 3, padding=1), nn.GELU(),
                nn.Conv2d(c, 1, 1),
            )

        def _feature_map(self, mel: torch.Tensor) -> torch.Tensor:
            return self.encoder.body(self.encoder.stem(mel))     # (B, 128, m', f')

        def forward(self, wav: torch.Tensor, mask: torch.Tensor):
            mel = self.frontend(wav)                             # (B,1,M,F), standardised
            corrupted = mel * (1.0 - mask)                       # zero-fill masked patches
            feat = self._feature_map(corrupted)
            rec = self.decoder(feat)
            rec = F.interpolate(rec, size=mel.shape[-2:], mode="bilinear", align_corners=False)
            # MAE loss: MSE on masked region only
            denom = mask.sum().clamp_min(1.0)
            loss = (((rec - mel) ** 2) * mask).sum() / denom
            return loss, rec, mel

    return MaskedSpecAutoencoder()


def export_encoder_state(ssl_model, path):
    """Save the pretrained ``AudioCNNEncoder`` weights for A1 fine-tuning."""
    from pathlib import Path
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ssl_model.encoder.state_dict(), path.as_posix())
    return path
