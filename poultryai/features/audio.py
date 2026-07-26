"""Audio front-end for respiratory-disease acoustics (cough / rale / sneeze).

We follow the current consensus in poultry respiratory acoustics: convert the waveform
to a log-mel spectrogram and let a neural encoder learn from the time-frequency image
(Audio Spectrogram Transformer / CNN family). Log-mel is a strong, well-understood
default; a wavelet-scattering alternative is provided as an option because WST+LSTM is
competitive on chicken vocalisations in recent work.

References
----------
* Gong et al., 2021, "AST: Audio Spectrogram Transformer".
* SmartEars (2025) and ASCT-CC (2026): spectrogram-based chicken cough detection.
* WST+LSTM on chicken vocalisations (2024/2025).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio  # type: ignore
import torchaudio.transforms as T  # type: ignore


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    n_fft: int = 1024
    hop_length: int = 256          # 16 ms hop -> ~62.5 frames/s
    n_mels: int = 64
    f_min: float = 50.0            # cough/rale energy is low-mid band; drop sub-bass hum
    f_max: float = 8000.0
    top_db: float = 80.0
    # SpecAugment (train-time only)
    freq_mask: int = 12
    time_mask: int = 24


class LogMelFrontend(torch.nn.Module):
    """Waveform -> log-mel spectrogram, with optional SpecAugment in train mode.

    Output shape: (B, 1, n_mels, n_frames) — a single-channel image for the encoder.
    Kept as an ``nn.Module`` (not a preprocessing script) so it exports cleanly with the
    model to ONNX/TorchScript for the RPi4 edge.
    """

    def __init__(self, cfg: AudioConfig = AudioConfig(), augment: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.augment = augment
        self.melspec = T.MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=cfg.n_mels, f_min=cfg.f_min, f_max=cfg.f_max, power=2.0,
        )
        self.to_db = T.AmplitudeToDB(stype="power", top_db=cfg.top_db)
        self.freq_mask = T.FrequencyMasking(cfg.freq_mask)
        self.time_mask = T.TimeMasking(cfg.time_mask)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, n_samples). Peak-normalise for level invariance across mics/gain.
        wav = wav / (wav.abs().amax(dim=-1, keepdim=True) + 1e-6)
        mel = self.melspec(wav)                 # (B, n_mels, n_frames)
        mel = self.to_db(mel)
        # Per-sample standardisation (CMVN-style) stabilises training across saloons.
        mel = (mel - mel.mean(dim=(-2, -1), keepdim=True)) / (
            mel.std(dim=(-2, -1), keepdim=True) + 1e-5)
        if self.augment and self.training:
            mel = self.time_mask(self.freq_mask(mel))
        return mel.unsqueeze(1)                  # (B, 1, n_mels, n_frames)
