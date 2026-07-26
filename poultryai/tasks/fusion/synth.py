"""Twin-synthetic multimodal flock generator (torch-free).

Produces ``schema.Window`` objects: env time series (from ``saloontwin`` when importable, else
a self-contained fallback), a raw audio waveform, vision behavioural indices, flock metadata, a
multi-label onset label, and the per-disease ``onset_offset`` the lead-time metric needs.

Disease realism is injected as **pre-onset signatures** that ramp up over a horizon before a
veterinary-confirmed onset time, differently per modality per disease — respiratory diseases
show up in **audio** (cough) and NH3/CO2 drift; enteric diseases show up in **vision**
(activity/feeding/ distribution decline). This is what lets the audio/vision ablations be
meaningful. It is **simulation** to exercise the C1 pipeline, not a scientific result.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from poultryai.data.schema import (DISEASES, Disease, ENV_CHANNELS, FlockMeta,
                                   N_DISEASES, VISION_CHANNELS, Window)

MIN = 60.0  # seconds per minute (env sampled 1/min)


# --------------------------------------------------------------------------------------
# Disease → per-modality pre-onset signature strengths (0..1)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DiseaseSignature:
    audio_cough: float = 0.0        # cough/rale burst intensity added to audio
    env_nh3: float = 0.0            # NH3 drift (ppm) coupling
    env_co2: float = 0.0            # CO2 drift
    env_temp_instab: float = 0.0    # thermoregulation instability
    vis_activity_drop: float = 0.0  # activity_index decline
    vis_distress: float = 0.0       # distress_index rise
    vis_feed_drop: float = 0.0      # feed_visit_rate decline
    vis_uniformity_drop: float = 0.0  # distribution_uniformity decline (piling/clumping)


# respiratory diseases → audio + air-quality; enteric → behaviour/vision
SIGNATURES: dict[Disease, DiseaseSignature] = {
    Disease.INFECTIOUS_BRONCHITIS: DiseaseSignature(
        audio_cough=0.9, env_nh3=0.5, env_co2=0.3, vis_activity_drop=0.4, vis_distress=0.5),
    Disease.AVIAN_INFLUENZA: DiseaseSignature(
        audio_cough=0.8, env_nh3=0.3, vis_activity_drop=0.7, vis_distress=0.8, vis_feed_drop=0.6),
    Disease.NEWCASTLE: DiseaseSignature(
        audio_cough=0.7, env_nh3=0.4, vis_activity_drop=0.6, vis_distress=0.6,
        vis_uniformity_drop=0.4),
    Disease.COCCIDIOSIS: DiseaseSignature(
        audio_cough=0.05, vis_activity_drop=0.6, vis_feed_drop=0.7, vis_distress=0.5,
        vis_uniformity_drop=0.5, env_nh3=0.2),
    Disease.COLIBACILLOSIS: DiseaseSignature(
        audio_cough=0.15, vis_activity_drop=0.5, vis_feed_drop=0.5, vis_distress=0.6,
        env_nh3=0.3),
}


@dataclass
class SynthConfig:
    n_flocks: int = 12
    healthy_frac: float = 0.4            # fraction of flocks with no disease
    flock_days: float = 4.0             # grow-out slice length
    window_minutes: int = 240
    stride_minutes: int = 60
    vision_cadence_min: int = 10        # one vision index vector every N minutes
    audio_seconds: float = 4.0
    sample_rate: int = 16_000
    pre_onset_horizon_h: float = 48.0   # signature ramp starts this long before onset
    label_horizon_h: float = 72.0       # windows within this pre-onset horizon are positive
    seed: int = 42
    use_twin: bool = True


# --------------------------------------------------------------------------------------
# Environment base (twin if available, else fallback)
# --------------------------------------------------------------------------------------
def _locate_twin():
    twin_dir = Path(__file__).resolve().parents[4] / "digital-twin"
    if twin_dir.exists() and str(twin_dir) not in sys.path:
        sys.path.insert(0, str(twin_dir))
    try:
        from saloontwin import SaloonTwin, SetpointController, SimConfig  # type: ignore
        from saloontwin.config import Weather  # type: ignore
        return SaloonTwin, SetpointController, SimConfig, Weather
    except Exception:
        return None


def _env_from_twin(n_min: int, age0: float, t_out: float, rng: np.random.Generator):
    got = _locate_twin()
    if got is None:
        return None
    SaloonTwin, SetpointController, SimConfig, Weather = got
    cfg = SimConfig(start_age_days=age0, weather=Weather(t_out_c=t_out))
    twin = SaloonTwin(cfg, seed=int(rng.integers(1 << 30)))
    ctrl = SetpointController(band=1.0)
    obs = twin.observe(noise=True)
    temp = np.empty(n_min); rh = np.empty(n_min); co2 = np.empty(n_min); nh3 = np.empty(n_min)
    for i in range(n_min):
        twin.step(ctrl.act(obs))
        obs = twin.observe(noise=True)
        temp[i], rh[i], co2[i], nh3[i] = obs["temp_c"], obs["humidity_pct"], obs["co2_ppm"], obs["nh3_ppm"]
    return temp, rh, co2, nh3


def _env_fallback(n_min: int, age0: float, rng: np.random.Generator):
    """OU-ish fallback around plausible setpoints if the twin is unavailable."""
    target_t = 24.0 - 0.15 * max(0.0, age0 - 21)
    def ou(x0, mean, theta, sigma):
        x = np.empty(n_min); x[0] = x0
        for i in range(1, n_min):
            x[i] = x[i-1] + theta * (mean - x[i-1]) + sigma * rng.standard_normal()
        return x
    temp = ou(target_t, target_t, 0.02, 0.05)
    rh = np.clip(ou(60, 62, 0.02, 0.2), 30, 90)
    co2 = np.clip(ou(1200, 1300, 0.01, 8), 420, 4000)
    nh3 = np.clip(ou(8, 12, 0.005, 0.1), 0, 40)
    return temp, rh, co2, nh3


def _light_program(n_min: int) -> np.ndarray:
    """Simple 18L:6D photoperiod in lux (day ~20 lux, night ~5 lux)."""
    minutes = np.arange(n_min)
    hour = (minutes / 60.0) % 24.0
    return np.where(hour < 18.0, 20.0, 5.0).astype(np.float64)


# --------------------------------------------------------------------------------------
# Signature ramp
# --------------------------------------------------------------------------------------
def _ramp(n_min: int, onset_min: float | None, horizon_h: float) -> np.ndarray:
    """0 before the horizon, rising linearly to 1 at onset, staying 1 after (clinical)."""
    if onset_min is None:
        return np.zeros(n_min)
    t = np.arange(n_min, dtype=np.float64)
    h_min = horizon_h * 60.0
    r = (t - (onset_min - h_min)) / h_min
    return np.clip(r, 0.0, 1.0)


# --------------------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------------------
def _synth_audio(cough_level: float, sr: int, seconds: float, rng: np.random.Generator) -> np.ndarray:
    n = int(seconds * sr)
    # farm background: low-pass-ish noise
    bg = np.cumsum(rng.standard_normal(n)) * 0.002
    bg = (bg - bg.mean()) * 0.3
    sig = bg + 0.02 * rng.standard_normal(n)
    if cough_level > 0.01:
        n_cough = rng.poisson(2.5 * cough_level)     # more coughs when sicker
        for _ in range(int(n_cough)):
            start = int(rng.uniform(0, n - sr // 4))
            dur = int(sr * rng.uniform(0.05, 0.2))
            env = np.exp(-np.linspace(0, 6, dur))
            burst = env * rng.standard_normal(dur) * (0.4 + 0.6 * cough_level)
            sig[start:start + dur] += burst
    return np.clip(sig, -1.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------------------
# One flock
# --------------------------------------------------------------------------------------
def generate_flock(cfg: SynthConfig, flock_idx: int, disease: Disease | None,
                   rng: np.random.Generator) -> list[Window]:
    n_min = int(cfg.flock_days * 24 * 60)
    age0 = float(rng.uniform(10, 30))
    t_out = float(rng.uniform(0, 26))

    base = _env_from_twin(n_min, age0, t_out, rng) if cfg.use_twin else None
    if base is None:
        base = _env_fallback(n_min, age0, rng)
    temp, rh, co2, nh3 = (np.array(x, dtype=np.float64) for x in base)
    lux = _light_program(n_min)

    onset_min = None
    sig = DiseaseSignature()
    if disease is not None:
        # onset in the middle-to-late part of the slice so pre-onset windows exist
        onset_min = float(rng.uniform(n_min * 0.5, n_min * 0.9))
        sig = SIGNATURES[disease]

    ramp = _ramp(n_min, onset_min, cfg.pre_onset_horizon_h)
    # env overlay (respiratory drift)
    nh3 = nh3 + 12.0 * sig.env_nh3 * ramp
    co2 = co2 + 700.0 * sig.env_co2 * ramp
    temp = temp + sig.env_temp_instab * 1.5 * ramp * rng.standard_normal(n_min)

    env_series = np.stack([temp, rh, lux, nh3, co2], axis=1)          # (n_min, 5), ENV order

    # vision indices at coarse cadence, degrading with the ramp
    vcad = cfg.vision_cadence_min
    v_idx = np.arange(0, n_min, vcad)
    vramp = ramp[v_idx]
    activity = np.clip(0.5 - 0.4 * sig.vis_activity_drop * vramp + 0.03 * rng.standard_normal(len(v_idx)), 0, 1)
    uniformity = np.clip(0.8 - 0.5 * sig.vis_uniformity_drop * vramp + 0.03 * rng.standard_normal(len(v_idx)), 0, 1)
    piling = np.clip(0.1 + 0.6 * sig.vis_uniformity_drop * vramp + 0.03 * rng.standard_normal(len(v_idx)), 0, 1)
    feed = np.clip(0.4 - 0.3 * sig.vis_feed_drop * vramp + 0.03 * rng.standard_normal(len(v_idx)), 0, 1)
    distress = np.clip(0.1 + 0.7 * sig.vis_distress * vramp + 0.03 * rng.standard_normal(len(v_idx)), 0, 1)
    vis_full = np.stack([activity, uniformity, piling, feed, distress], axis=1)  # VISION order

    meta = FlockMeta(owner=f"synthfarm", saloon=f"s{flock_idx:02d}",
                     age_days=int(age0), stocking_density=float(rng.uniform(12, 18)),
                     mortality_rate_24h=float(0.001 + 0.01 * (0 if disease is None else 1)))

    windows: list[Window] = []
    W = cfg.window_minutes
    for end in range(W, n_min + 1, cfg.stride_minutes):
        env_w = env_series[end - W:end]
        # vision timesteps whose minute index falls in [end-W, end)
        vm = (v_idx >= end - W) & (v_idx < end)
        vision_w = vis_full[vm] if vm.any() else vis_full[:1]
        strength = ramp[end - 1]
        audio_w = _synth_audio(sig.audio_cough * strength, cfg.sample_rate,
                               cfg.audio_seconds, rng)

        labels = np.zeros(N_DISEASES, dtype=np.float32)
        onset_offset = np.full(N_DISEASES, np.inf, dtype=np.float32)
        if disease is not None:
            d = DISEASES.index(disease)
            offset_s = (onset_min - (end - 1)) * MIN           # +future / -past
            onset_offset[d] = np.float32(offset_s)
            # positive if within the labelling horizon before onset, or post-onset (clinical)
            if offset_s <= cfg.label_horizon_h * 3600.0:
                labels[d] = 1.0

        windows.append(Window(
            t_start=float((end - W) * MIN), t_end=float((end - 1) * MIN),
            env=env_w.astype(np.float32), audio=audio_w,
            vision=vision_w.astype(np.float32), meta=meta,
            labels=labels, onset_offset=onset_offset,
            sample_id=f"{meta.saloon}:{end}"))
    return windows


def generate_dataset(cfg: SynthConfig | None = None) -> list[Window]:
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    n_healthy = int(round(cfg.n_flocks * cfg.healthy_frac))
    diseases = [None] * n_healthy + [
        DISEASES[i % N_DISEASES] for i in range(cfg.n_flocks - n_healthy)]
    rng.shuffle(diseases)
    windows: list[Window] = []
    for k, dz in enumerate(diseases):
        windows.extend(generate_flock(cfg, k, dz, rng))
    return windows
