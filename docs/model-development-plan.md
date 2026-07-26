# Model Development Plan — what to build next (script → train → test → deploy)

## The key decision
The **multimodal onset model** (`poultryai/models/multimodal.py`, `scripts/train.py`)
is the thesis centrepiece, but it **cannot be trained yet**: it needs time-aligned
env+audio+vision windows with veterinary **onset times** (see `data_dictionary.md`),
which only real longitudinal farm data (or the digital twin) provides. `load_windows`
raises on purpose so we never "train on nothing".

**So build the single-modality *component* models first** — each uses data we already
have, is independently useful, produces the encoder the fusion model later reuses, and
proves the full **script → train → test → deploy** pipeline end-to-end.

## Recommended order

| # | Model | Data (have) | Backbone | Feeds later |
|---|---|---|---|---|
| **1 (next)** | **Fecal-image disease classifier** | Zenodo AI4D (4 classes) | CNN transfer (ResNet18/EffNet-B0) | standalone edge model; covers Coccidiosis + Newcastle |
| 2 | Audio health / cough classifier | Mendeley vocalization | log-mel → small CNN (reuse `features/audio.py`, `models/audio_encoder.py`) | the **audio encoder** of the fusion model |
| 3 | Vision detector → behaviour indices | Roboflow (YOLO) | YOLOv8 + tracker | the **vision indices** the fusion model consumes |
| 4 | SSL pretraining (masked env / audio-MAE) | unlabeled logs + twin | — | initializes encoders for small labelled sets |
| 5 | **Multimodal fusion** (the centrepiece) | real/twin longitudinal | existing package | the thesis early-warning claim |
| 6 | Control (MPC / RL) | digital twin | `saloontwin` | the C3 closed-loop result |

Start with **#1** — smallest, fully-labelled, and it builds the reusable MLOps spine.

---

## Model #1 — fecal-image disease classifier (do this now)

**Placement (keep the multimodal package clean):** a self-contained task module
`poultryai/tasks/fecal/` + `configs/fecal.yaml` + `scripts/train_fecal.py`, reusing
`utils/config`, `utils/seed`, the MLflow tracker, and the ONNX-export pattern from
`inference/export_edge.py`.

### Script (data)
- `torchvision.datasets.ImageFolder` over the Zenodo class folders
  (`Healthy/ Coccidiosis/ Salmonella/ Newcastle/`).
- **Stratified split**, but ⚠️ **check for leakage**: if several images come from the
  same bird/sampling session, split by that group id — otherwise test scores inflate
  (the classic livestock-CV mistake, already flagged in `methodology.md`).
- Transforms: resize 224², train-time augment (flip, rotate, colour jitter),
  normalize with ImageNet stats.

### Train
- Backbone **pretrained** (ResNet18 or EfficientNet-B0 via `torchvision`/`timm`),
  replace the head with a 4-logit classifier; fine-tune.
- Loss: **cross-entropy with class weights** (imbalance). Optimizer AdamW + cosine,
  mixed precision, early stop on **val macro-F1**. Log to **MLflow**.
- `python scripts/train_fecal.py --config configs/fecal.yaml`.

### Test
- Held-out (leakage-safe) test set. Report **per-class precision/recall/F1, macro-F1,
  AUROC, confusion matrix**, and **ECE** (calibration) — consistent with the thesis'
  metric philosophy (`metrics.md`). Save the report + confusion matrix as artifacts.

### Deploy
- Export to **ONNX** (`inference/export_edge.py` conventions) → run with
  **onnxruntime** on the RPi4/VPS ML service.
- Publish predictions to MQTT under a new **`ai/{owner}/{saloon}/fecal/{class,score}`**
  topic (add to the topic standard); Node-RED/Grafana/mobile app consume it.
- Deployment context: periodic **manure/dropping images** (fixed camera over the belt/
  litter, or operator upload) → classifier → early alert.

**Framing (honest):** this is a **component demonstrator** — a real, deployable model
covering 2 of the 5 diseases — **not** the thesis' early-warning lead-time claim, which
remains the multimodal onset model (#5).

---

## Set up once (shared MLOps spine — reused by every model)
- `pip install -e ".[dev,track]"` (torch, lightning, sklearn, mlflow, pytest, ruff).
- **MLflow** tracking server on the VPS (`logging.tracker: mlflow`); versioned runs +
  model registry.
- **CI:** `pytest` (`poultryai/tests`, twin smoke tests), `ruff`, and a smoke train
  on a tiny subset so the pipeline can't silently break.
- Keep everything **config-driven + seeded** (already the package's style).

## How the pieces converge
Component models #1–#3 give trained **encoders + a working pipeline**; #4 pretrains them
on cheap unlabeled data; once longitudinal data (real farm P0, or the **digital twin**)
exists, #5 fuses the encoders into the onset model, and #6 uses the twin to train/validate
the controller — the two headline results.

### Immediate action
Scaffold `poultryai/tasks/fecal/` + `scripts/train_fecal.py` + `configs/fecal.yaml`
(dataset, transfer-learning trainer, eval report, ONNX export). Then point
`Data/downloads/zenodo_.../` at it and run.
