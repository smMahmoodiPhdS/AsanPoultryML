"""C1 — Multimodal onset model, exercised end-to-end on **twin-synthetic** data.

The thesis centrepiece (`poultryai.models.multimodal.MultimodalEarlyWarning`) needs
time-aligned env+audio+vision windows with veterinary **onset times**, which only real
longitudinal farm data or the digital twin can provide. This package generates that data
*synthetically* — env trajectories from `saloontwin`, with per-disease **pre-onset signatures**
injected across modalities — so the whole **train → lead-time evaluation** pipeline (and the
audio/vision ablations) runs and is validated *before* real data exists.

Honest framing: this is **simulation**, to de-risk and prove the C1 pipeline, not a scientific
result. Absolute lead-time numbers are meaningful only once the model trains on real (or
twin-calibrated) data. What it demonstrates now is that the onset labelling, the flock-grouped
split, the model I/O contract, and the lead-time protocol are all correct and connected.

Modules
-------
* ``synth``       — generate ``schema.Window`` flocks (twin env + disease signatures). Torch-free.
* ``leadtime_io`` — turn windows + per-window scores into ``eval.leadtime.FlockSeries``. Torch-free.
"""
from __future__ import annotations

from .synth import SynthConfig, DiseaseSignature, generate_dataset, generate_flock

__all__ = ["SynthConfig", "DiseaseSignature", "generate_dataset", "generate_flock"]
