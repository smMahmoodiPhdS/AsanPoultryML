---
title: ml-models brief
type: brief
level: project
updated: 2026-09-11
---

# Brief — ml-models (`poultryai`)

**The scientific centre of the thesis.** Predicts onset of five broiler diseases
from **environmental time series + audio + vision**, and measures *how early* it
warns versus a human supervisor.

- Trains on workstation/GPU. Deploys to the **Raspberry Pi 4 edge** via ONNX.
- Own git repo. Moved from `Applications/ml-models` on 2026-09-11.

## Tasks

`poultryai/tasks/` — `audio/`, `vision/`, `fecal/`, `fecal_seg/`, `health_seg/`.
Each has a README and a built manifest CSV.

## Data paths after the move

Manifests and scripts reference the datasets **relative to this repo root**:
`../Data/downloads/...` (was `../../Data/...` before the move). Vault-root-relative
instructions now read `PhdThesis/Data/downloads/...`.

> ⚠️ **The Roboflow images are not on disk.**
> `PhdThesis/Data/downloads/Roboflow/Data/` is empty, so every path the vision
> manifests point at (`../Data/downloads/Roboflow/Data/downloads/<dataset>/...`)
> is missing, and `train_detector.py` falls back to the class name `object`.
> The paths are right; the export has to be re-downloaded. Pre-existing,
> unrelated to the 2026-09-11 move.

## Read

`docs/methodology.md` — design rationale and literature grounding ·
`docs/metrics.md` — the lead-time protocol · `docs/data_dictionary.md` ·
[[PhdThesis/ml-models/docs/ai-model-plan|ai-model-plan]] ·
[[PhdThesis/ml-models/docs/model-development-plan|model-development-plan]] ·
[[PhdThesis/Data/brief|Data brief]]
