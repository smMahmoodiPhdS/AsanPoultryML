"""Canonical data schemas for the multimodal poultry-disease dataset.

Everything downstream (feature extraction, datamodule, models, evaluation) depends on
these types, so the on-disk data contract is defined once, here, and validated.

Design choices
--------------
* We keep three *aligned* modality streams — environmental time series, audio, and
  vision-derived behavioural indices — plus flock metadata and a per-window label.
* Time is explicit and UTC-epoch; alignment across modalities is by window, not by
  assuming synchronous sampling (sensors, mic, and camera all sample differently).
* Labels support *onset* modelling: we store the veterinary-confirmed onset time so we
  can compute lead-time (how early we predicted) rather than only window accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class Disease(str, Enum):
    """The five target broiler diseases (see PhdThesis/Thesis)."""
    NEWCASTLE = "newcastle"
    AVIAN_INFLUENZA = "avian_influenza"
    INFECTIOUS_BRONCHITIS = "infectious_bronchitis"
    COCCIDIOSIS = "coccidiosis"
    COLIBACILLOSIS = "colibacillosis"


DISEASES: tuple[Disease, ...] = tuple(Disease)
N_DISEASES: int = len(DISEASES)

# Canonical environmental channel order. Never reorder — model checkpoints depend on it.
ENV_CHANNELS: tuple[str, ...] = (
    "temp_c", "humidity_pct", "lux", "nh3_ppm", "co2_ppm",
)
# Canonical vision behavioural-index channel order.
VISION_CHANNELS: tuple[str, ...] = (
    "activity_index", "distribution_uniformity", "piling_score",
    "feed_visit_rate", "distress_index",
)


@dataclass(slots=True)
class FlockMeta:
    """Static-per-window flock context. Concatenated to the fused embedding."""
    owner: str
    saloon: str
    age_days: int
    stocking_density: float          # birds / m^2
    breed: str = "ross308"
    mortality_rate_24h: float = 0.0  # rolling, as a weak prior

    def to_vector(self) -> np.ndarray:
        # Simple, explicit featurisation; categorical breed handled by the datamodule.
        return np.asarray(
            [self.age_days, self.stocking_density, self.mortality_rate_24h],
            dtype=np.float32,
        )


@dataclass(slots=True)
class Window:
    """One training/inference example: a fixed-length, multi-window-aligned sample.

    Shapes
    ------
    env      : (T_env, C_env)      e.g. (240, 5) for 4 h @ 1 sample/min
    audio    : (n_samples,)        raw mono waveform for the window (resampled to sr)
    vision   : (T_vis, C_vis)      behavioural indices per vision timestep
    labels   : (N_DISEASES,)       multi-label {0,1} — a window can carry >1 condition
    """
    t_start: float                 # epoch seconds (window start)
    t_end: float
    env: np.ndarray
    audio: np.ndarray
    vision: np.ndarray
    meta: FlockMeta
    labels: np.ndarray
    # onset_offset[d] = seconds from window END to confirmed onset of disease d.
    #   negative  -> onset already happened (post-onset window)
    #   positive  -> onset is in the future (pre-onset window; the early-warning regime)
    #   +inf      -> disease never occurs in this flock (healthy)
    onset_offset: np.ndarray = field(
        default_factory=lambda: np.full(N_DISEASES, np.inf, dtype=np.float32))
    sample_id: Optional[str] = None

    def __post_init__(self) -> None:
        assert self.env.ndim == 2 and self.env.shape[1] == len(ENV_CHANNELS), \
            f"env must be (T,{len(ENV_CHANNELS)}), got {self.env.shape}"
        assert self.vision.ndim == 2 and self.vision.shape[1] == len(VISION_CHANNELS), \
            f"vision must be (T,{len(VISION_CHANNELS)}), got {self.vision.shape}"
        assert self.labels.shape == (N_DISEASES,), "labels must be (N_DISEASES,)"
