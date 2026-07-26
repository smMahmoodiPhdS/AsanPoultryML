"""A1 — Acoustic health / abnormality classifier (the reusable audio encoder).

Trains on the Mendeley chicken-vocalization corpus (Healthy / Unhealthy / Noise, 48 kHz) to
produce the **audio encoder** the multimodal fusion model reuses. This is the modality that
addresses the respiratory-disease gap (Infectious Bronchitis, Avian Influenza, the respiratory
phase of Newcastle) that the image models cannot see — cough/rale/sneeze are audible before
visible decline (``docs/ai-model-plan.md`` A1).

Honest scope: the labels are a **binary health signal** (Healthy vs Unhealthy) plus an explicit
**Noise** (farm-background) class — so A1 is *acoustic health/abnormality detection*, not
per-disease acoustic diagnosis. The Noise class doubles as a vocalization-vs-background gate and
makes noise-mixing / SpecAugment realistic.

Reuses the package's edge-ready front-end and encoder:
* ``poultryai.features.audio.LogMelFrontend`` — waveform → log-mel (+ SpecAugment), inside the
  model so ONNX consumes raw audio.
* ``poultryai.models.audio_encoder.AudioCNNEncoder`` — the compact CNN that becomes the fusion
  model's audio branch.

Modules
-------
* ``windowing`` — split a clip into fixed windows (short-clip pad, per-file cap). Torch-free.
* ``splits``    — scan wavs (``wave`` stdlib), recording-level stratified split. Torch-free.
* ``data``      — load + resample 48k→16k + window (needs torch/torchaudio or soundfile).
* ``model``     — front-end + encoder + 3-logit head; ONNX export from raw waveform.
* ``eval``      — window- and recording-level macro-F1, AUROC, ECE, confusion.
"""
from __future__ import annotations

CLASSES: tuple[str, ...] = ("healthy", "unhealthy", "noise")
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

# source folder name (Mendeley) → canonical class
FOLDER_TO_CLASS: dict[str, str] = {"Healthy": "healthy", "Unhealthy": "unhealthy",
                                   "Noise": "noise"}
