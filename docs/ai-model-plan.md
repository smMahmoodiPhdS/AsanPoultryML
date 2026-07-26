# AI Model Plan — Vision / Audio / Control (data-grounded, PhD-level)

*Companion to [`methodology.md`](methodology.md), [`metrics.md`](metrics.md),
[`data_dictionary.md`](data_dictionary.md), [`model-development-plan.md`](model-development-plan.md)
and [[Docs/Completion-Roadmap-and-AI-Strategy]]. This document turns those into a concrete,
**dataset-grounded** build plan for the three model pillars, using the data now on disk in
`Data/downloads/`.*

Status: **in implementation** · Updated: 2026-07-21 · Author: research plan (amend freely).
All component encoders (V1–V4, A1–A2), the control pillar (C-1/C-2/C-3), and the C1 fusion
pipeline are now **built, tested, and runnable** — see the implementation status in §0.5.

---

## 0. What this document adds

The methodology and roadmap already fix the *science* (onset-aware multimodal early warning +
closed-loop control) and the *evaluation* (lead-time @ fixed FPR, ablations, twin→on-farm A/B).
What was missing was a plan tied to the **actual data we downloaded**. This document:

1. Inventories the four downloaded datasets **as they really are on disk** (verified, §1), with
   the honest caveats (dataset overlap, disease coverage gaps, leakage risk).
2. Specifies the **vision, audio and control models** each dataset supports — architecture,
   training regime, evaluation, edge-export, and the reusable encoder each yields (§3–§5).
3. States the **cross-cutting protocols** (leakage-safe splits, imbalance, calibration,
   uncertainty/OOD, MLOps) that make the results defensible (§6).
4. Maps every model to the thesis contributions **C1/C2/C3**, the **five target diseases**, and
   the thesis chapters (§7–§8), with a phased milestone plan and risk register (§9–§10).

Guiding principle (unchanged): the component models are trained on data we have *now*; the
multimodal **onset** model and the **control** A/B need longitudinal real data or the digital
twin, so they are sequenced last and de-risked by SSL pretraining + `saloontwin`.

---

## 0.5 Implementation status (built, 2026-07-21)

Every buildable component across the three pillars is now scaffolded, unit-tested, and runnable.
Each carries a **leakage-safe data gate** built and verified on the real data; model training
needs the GPU workstation (torch/torchaudio/ultralytics), but all torch-free gates, generators,
metrics, and the control + fusion pipelines run and are tested now. **59 tests pass** (46
ml-models + 9 control + 4 twin smoke).

| ID | Component | Location | Data gate (verified) | Tests |
|---|---|---|---|---|
| **V1** | Fecal disease classifier | `poultryai/tasks/fecal/` | md5 dedup + stratified group split; 277 dups collapsed, 6 label-conflicts dropped → 5662/1203/1196, leakage-free | 6 |
| **V2** | Fecal lesion segmentation | `poultryai/tasks/fecal_seg/` | **inherits V1 split by filename** (Zenodo↔Kaggle); 6811/6812 matched; seg-test ∩ V1-train = 0 | 5 |
| **V3** | Detection→tracking→behaviour indices | `poultryai/tasks/vision/` | clip-aware re-split (audited shipped-split leakage: 15/39/3/11 clips); numpy tracker + 5 `VISION_CHANNELS` extractor | 9 |
| **V4** | Healthy/Sick posture segmentation | `poultryai/tasks/health_seg/` | reuses V3 clip-aware `healthy_sick` split; YOLOv8-seg mask rasteriser | 7 |
| **A1** | Acoustic health classifier (audio encoder) | `poultryai/tasks/audio/` | recording-level split (242/52/52); windowing + per-file cap | 8 |
| **A2** | SSL audio pretraining (masked-spectrogram) | `poultryai/tasks/audio/ssl.py` | pretrain on train+val only (never test); patch-mask + label-efficiency study | 6 |
| **C-1** | Ventilation/thermal MPC (solver-free) | `digital-twin/saloontwin/control/mpc.py` | sampling/CEM MPC; twin-in-the-loop rollout | — |
| **C-2** | Learned controller (ES) + RL env | `.../control/es_policy.py`, `env.py` | evolution-strategy policy trains in-twin; gym wrapper for SAC | — |
| **C-3** | A/B harness + safety shield | `.../control/{ab_eval,shield,scenarios}.py` | paired KPIs across scenarios×seeds; hard interlocks | 9 |
| **C1** | Multimodal onset model + lead-time | `poultryai/tasks/fusion/` | twin-synthetic flock generator; **lead-time protocol runs end-to-end** | 5 |

**Runnable-now highlights (no GPU):** every leakage-safe manifest builds and is committed; the
V3 tracker + behavioural-index extraction runs on real Roboflow frames; the control A/B shows
the learned/MPC controllers beat the baseline hysteresis (lower NH₃, 0 shield overrides vs 3.6 %)
and the ES policy trains in ~1 s; the C1 fusion generator + lead-time protocol run end-to-end in
~3 s (all diseases detected before onset on the signature-proxy oracle).

**What remains (needs data/hardware, not code):** GPU training of every model; real farm logging
(roadmap P0) for onset labels; setpoint-table + twin-physics calibration; then swap synthetic
windows for real ones and run the on-farm control A/B — the two headline results.

---

## 1. Dataset inventory (verified on disk, 2026-07-21)

Located under `Data/downloads/`. Counts and formats below were checked directly, not assumed.

### 1.1 Fecal disease imagery — enteric/systemic diseases

| Source | Content | Classes (file counts) | Format |
|---|---|---|---|
| `Kaggle/kaggle_chicken_disease/` | classification images (flat, class-prefixed filenames) + `train_data.csv` | Coccidiosis 2476, Healthy 2404, Salmonella 2625, Newcastle 562 — plus PCR-labeled subsets (`pcrcocci/pcrhealthy/pcrncd/pcrsalmo`, ~373 each) | 8 067 JPG |
| `zenodo_4628934_poultry_fecal/` | **LabelMe segmentation JSON** with embedded base64 image + one polygon per image | cocci 2103, healthy 2057, salmo 2276, ncd 376 | 6 812 JSON (~15 GB; each JSON carries `imageData`, `imageHeight/Width`, and a `shapes[].points` polygon labeled with the disease) |

**Critical facts.** These two are the **same source dataset** (the public "poultry disease
diagnostics" fecal corpus) in two representations: Kaggle = raw classification JPGs; Zenodo =
the **segmentation masks** (polygons) with the image embedded in the JSON. Treat them as **one
dataset with two label types**, not two independent datasets — combining them naively would
leak identical images across train/test.

**Disease coverage vs. the thesis five** (`Disease` enum: Newcastle, Avian Influenza,
Infectious Bronchitis, Coccidiosis, Colibacillosis): images cover **Coccidiosis ✓** and
**Newcastle ✓** directly, plus **Salmonella** (an extra enteric class, *not* one of the five —
do not relabel it as Colibacillosis; they are different pathogens). **Avian Influenza,
Infectious Bronchitis and Colibacillosis are absent from all image data** — by design they are
the target of the **audio** modality (respiratory signs) and the multimodal onset model. So the
fecal work is honestly a *2-of-5 component demonstrator* (plus Salmonella), exactly as framed in
`model-development-plan.md`.

**Class imbalance:** Newcastle is the minority everywhere (~5–7×fewer than the majority classes)
— drives the weighted-loss / resampling choices in §3.

### 1.2 Vocalization audio — respiratory / general-health acoustics

`mendeley_vocalization/Chicken_Audio_Dataset/` — **48 kHz, mono, 16-bit WAV**, variable length
(~20–85 s clips): **Healthy 139, Unhealthy 121, Noise 86** (346 files, ~1.1 GB).

This is a **binary health signal** (healthy vs. unhealthy) plus an explicit **Noise** class
(farm-background) — ideal for a real-world-robust health classifier and, more importantly, for
pretraining the **audio encoder** that the fusion model reuses. It is *not* disease-labeled, so
it supports "acoustic health/abnormality detection," not per-disease acoustic diagnosis. The
Noise class is valuable: it lets us train a background-vs-vocalization gate and makes SpecAugment
+ noise-mixing augmentation realistic.

### 1.3 Vision detection / behaviour — Roboflow YOLO datasets

All in `Roboflow/Data/downloads/`, standard YOLO `train/valid/test` + `data.yaml`:

| Dataset | Task | Classes | Images (train/val/test) |
|---|---|---|---|
| `roboflow_broiler_detection` | bbox detection | `Chicken-Birds` (1) | 390 / 20 / 10 |
| `roboflow_tracking` | bbox detection (for tracking) | `rooster` (1) | 128 / 38 / 19 |
| `roboflow_chicken_pose` | **behaviour** bbox | `eat-drink`, `moving`, `rest` (3) | 1346 / 188 / 177 |
| `roboflow_healthy_sick` | **instance segmentation** (YOLOv8-seg polygons, ~27 pts) | `Healthy`, `Sick` (2) | 444 / 38 / 23 |

These are **small** (hundreds of images) → use for **transfer learning / fine-tuning** of
pretrained detectors, never training from scratch, and expect the main scientific use to be
**behavioural-index extraction**, not standalone SOTA detection. The pose/behaviour dataset is
the most valuable: it directly supports the `activity_index` / feeding-visit indices the vision
encoder consumes.

### 1.4 One-line gap statement

We can build **all three component encoders now** (vision-disease, vision-behaviour, audio-health).
We **cannot** yet build the multimodal onset model or the control A/B — those need time-aligned
longitudinal env+audio+vision with veterinary onset times (real farm P0) or the twin. Everything
below is sequenced around that reality.

---

## 2. How the data maps to the thesis contributions

| Contribution                                          | What it needs                             | Status (2026-07-21)                                    | This plan's models                                    |
| ----------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------- |
| **C1 — Multimodal early-warning + lead-time**         | aligned env+audio+vision with onset times | 🟢 **pipeline built & verified on twin-synthetic** data; awaits real onset labels for results | fusion model (§7), fed by V/A encoders below          |
| **C2 — Graceful robustness on messy farms**           | sensor-loss ablations, calibration, OOD   | 🟢 **modality-dropout + `--ablation` built**; ECE/OOD hooks in place | modality-dropout fusion, calibration/OOD heads (§6.3) |
| **C3 — Closed-loop disease-aware control (headline)** | calibrated twin + policy + safety shield  | 🟢 **twin A/B built & runs** (MPC + ES beat baseline, shield enforced); awaits calibration + on-farm A/B | MPC + RL-in-twin (§5)                                 |

Component encoders (V1–V4, A1–A2) are **independently publishable demonstrators** *and* the
initialisation for C1; the control pillar is the headline C3. This layering is what makes the
thesis defensible even if gold onset labels stay scarce.

---

## 3. Vision pillar

Four models, in dependency order. All export to ONNX for RPi4/VPS serving and publish to
`ai/{owner}/{saloon}/...` per the roadmap topic extension.

### V1 — Fecal disease classifier *(build first; the MLOps spine)*

- **Data:** `Kaggle` JPGs via `ImageFolder`-style loader, 4 classes (Coccidiosis, Healthy,
  Salmonella, Newcastle). Reconcile the `train_data.csv` labels with filename prefixes; keep the
  PCR-labeled subset as a **higher-confidence validation slice** (PCR-confirmed ground truth) —
  a nice robustness check (train on visual labels, verify on PCR-confirmed).
- **Backbone:** pretrained **EfficientNet-B0** or **ResNet-18** (torchvision/`timm`), head → 4
  logits, fine-tune. Also benchmark a **ConvNeXt-Tiny / MobileViT** for the accuracy–edge
  trade-off table (a small ablation the thesis can report).
- **Loss/opt:** class-weighted (or focal) cross-entropy for the Newcastle minority; AdamW +
  cosine, mixed precision, early-stop on **val macro-F1**. Augment: resize 224², flip, rotate,
  colour jitter, ImageNet norm.
- **Eval:** leakage-safe test set (§6.1); per-class P/R/F1, macro-F1, AUROC, confusion matrix,
  **ECE**. Report the PCR-slice separately.
- **Edge/serve:** ONNX + onnxruntime; publish `ai/{owner}/{saloon}/fecal/{class,score}`.
- **Thesis role:** component demonstrator covering 2-of-5 diseases + Salmonella; proves
  script→train→test→deploy end-to-end and the reusable MLOps/registry/ONNX pattern.

### V2 — Fecal lesion segmentation *(adds spatial explainability)*

- **Data:** `Zenodo` LabelMe JSON → decode base64 image + polygon mask per class. Same disease
  taxonomy as V1, so **the same group/leakage discipline applies** and V1/V2 must share the
  train/test partition (never test V2 on an image V1 trained on, and vice-versa).
- **Model:** **U-Net / DeepLabV3+ (ResNet backbone)** or **YOLOv8-seg** for lesion/region
  segmentation; alternatively a **segmentation-guided classifier** (mask as attention prior).
- **Why it matters scientifically:** turns V1 from a black-box classifier into an
  **explainable** one ("flagged on this lesion region"), directly supporting the roadmap's
  explainability requirement and vet trust (§2.1 of the roadmap). Also enables mask-based
  augmentation for V1.
- **Eval:** mIoU / Dice per class + downstream effect on V1 macro-F1; qualitative overlays.

### V3 — Detection → tracking → behavioural indices *(feeds the fusion vision encoder)*

- **Data:** `roboflow_broiler_detection` + `roboflow_tracking` (detection), `roboflow_chicken_pose`
  (behaviour states).
- **Pipeline:** **YOLOv8/YOLO-family** detector (fine-tuned on broiler_detection) → **ByteTrack/
  BoT-SORT** multi-object tracker → per-saloon **behavioural indices** matching
  `schema.VISION_CHANNELS`: `activity_index` (motion/speed), `distribution_uniformity` &
  `piling_score` (spatial spread / clustering), `feed_visit_rate` (visits to feeder ROI, aided by
  the pose `eat-drink` class), `distress_index` (proxy from posture/immobility).
- **Behaviour classifier:** fine-tune on `roboflow_chicken_pose` (eat-drink / moving / rest) to
  convert detections into activity/feeding statistics.
- **Edge-vs-VPS:** decide per ADR-002; small models (YOLOv8-n) can run RPi4, heavier on VPS.
  Privacy: **store indices, not raw video** where possible.
- **Thesis role:** produces the exact vision feature vector the fusion model's MLP consumes —
  i.e. this is the upstream that makes the "vision modality" real. Data is small, so frame the
  detector honestly as *transfer-learned* and validate index **stability**, not COCO-level mAP.
- **Leakage caution:** Roboflow frames are often consecutive **video frames** → split by
  **source clip/sequence**, not by random frame, or detection/tracking metrics inflate.

### V4 — Healthy/Sick posture segmentation *(a second, behaviour-side health cue)*

- **Data:** `roboflow_healthy_sick` YOLOv8-seg polygons (Healthy/Sick).
- **Model:** YOLOv8-seg fine-tune → a **posture/appearance health flag** complementary to the
  fecal (V1) and acoustic (A1) cues; contributes an additional `distress_index` signal.
- **Thesis role:** demonstrates a vision-based health signal that is **disease-agnostic** —
  useful as an early, cheap alarm and as an ablation ("vision-only") input to fusion.

---

## 4. Audio pillar

### A1 — Acoustic health / abnormality classifier *(the audio encoder)*

- **Data:** `mendeley_vocalization` (Healthy / Unhealthy / Noise), 48 kHz mono.
- **Front-end:** reuse `poultryai/features/audio.py` — resample 48 kHz → **16 kHz**, log-mel
  spectrogram, **SpecAugment**; window the long clips into fixed **4 s** frames (matches
  `data.audio_seconds`), keeping the front-end *inside* the model so ONNX consumes raw waveform.
- **Model:** compact **depthwise-separable CNN** (edge) reusing `models/audio_encoder.py`;
  benchmark against an **AST** (Audio Spectrogram Transformer) for the server model — the
  accuracy/size ablation for the thesis.
- **Task framing:** 3-class (Healthy / Unhealthy / Noise) — the Noise class doubles as a
  **vocalization-vs-background gate** and makes noise-mixing augmentation realistic; report
  binary health metrics too (Healthy vs Unhealthy).
- **Loss/opt/eval:** class-weighted CE, AdamW+cosine; macro-F1, AUROC, ECE; **clip-grouped
  split** (windows from the same recording must not straddle train/test — the audio analogue of
  §6.1).
- **Thesis role:** yields the **audio encoder** the fusion model reuses (§7). Targets the
  **respiratory** disease gap (IB/AI/ND respiratory phase) that images cannot see — the
  scientific justification for audio as a first-class modality.

### A2 — Self-supervised audio pretraining *(highest-leverage for small data)*

- **Data:** the unlabeled/weakly-labeled Mendeley clips now; unlabeled farm audio later (P0
  campaigns via the wired edge, since BLE cannot carry raw audio — ADR-010).
- **Method:** **Audio-MAE / masked-spectrogram reconstruction** (or contrastive SSL) to
  pretrain the audio encoder, then fine-tune A1 on the few labels. This is the roadmap's §1.2
  "make small labeled sets usable" lever and a genuine methodological contribution for a
  low-resource farm setting.
- **Deliverable:** a pretrained checkpoint that measurably improves A1 macro-F1 vs.
  from-scratch (report the delta and label-efficiency curve).

---

## 5. Control pillar (headline C3)

Built on the existing `Applications/digital-twin/saloontwin` (thermal / RH / CO₂ / NH₃ / Gompertz
growth balances, gym-like `reset`/`step`, baseline `SetpointController`, smoke tests).

### Layered controller architecture (reflex → predictive → risk-aware)

```
Safety reflex (node/edge)     hysteresis + interlocks — hard constraint, never overridden
   │
Predictive control (edge/VPS) MPC / RL setpoint optimisation (thermal, ventilation, light)
   │
Risk coupling (VPS)           disease-risk & AQ forecasts bias the controller's objective
```

### C-0 — Twin calibration *(prerequisite; unblocks everything below)*

- Fill the **age-banded setpoint table** (Ross 308 / Cobb 500 Aviagen handbook — cited) that the
  twin and labeling policy both depend on (roadmap §1.3, currently ❌ empty).
- Calibrate `saloontwin/physics.py` / `config.py` constants against **real InfluxDB logs** once
  P0 logging starts; until then treat outputs as *illustrative* and report twin results as
  simulation, clearly labeled.

### C-1 — Minimum-ventilation / thermal MPC *(the interpretable baseline AI controller)*

- **Problem:** hold NH₃/CO₂/RH within age-banded limits **and** temperature in band **while
  minimising heating energy** — the classic winter trade-off (fresh air clears AQ but loses heat).
- **Method:** receding-horizon **MPC** over the twin dynamics; objective = weighted (AQ-violation
  + thermal-violation + energy); constraints = actuator limits + the safety-shield interlocks as
  **hard** constraints. Interpretable, constraint-aware, safe → the right *baseline AI* controller.
- **Inputs:** short-horizon **NH₃/CO₂ forecasts** (small time-series head) feed the MPC.

### C-2 — Learned controller (RL-in-twin) *(the comparison for C3)*

- **Method:** train **RL** (e.g. SAC/PPO in the twin, or **offline/CQL** from logged trajectories)
  with a **safety shield** projecting actions into the interlock-feasible set. Compare vs. MPC and
  vs. the conventional `SetpointController`.
- **Disease-aware coupling (the novel bit):** the controller consumes the perception layer's
  **disease-risk score** (esp. respiratory, coupled to NH₃) and pre-emptively raises ventilation
  when risk rises — *detection→action*, the headline "AI agent for full-scope automation" claim.

### C-3 — Evaluation (twin → on-farm A/B)

- **Twin:** AI controller vs. baseline over many seeded episodes/weather profiles — AQ-violation
  hours, temperature-in-band %, **energy use**, and (via a coupled risk model) simulated disease
  incidence. Paired stats, ≥5 seeds, 95% CIs (matches `metrics.md` philosophy).
- **On-farm:** small **A/B trial** measuring disease incidence, mortality, **FCR**, energy,
  welfare proxies. This is the thesis' headline result.
- **Safety:** the node/edge hysteresis + interlocks are the final authority and are **never**
  overridden — encoded as a hard constraint layer, stated as an invariant.

---

## 6. Cross-cutting protocols (what makes it PhD-defensible)

### 6.1 Leakage-safe splitting *(the single most important discipline)*

- **Fecal V1/V2:** the same source images appear in both Kaggle and Zenodo → build **one master
  image index with content hashes**, dedup, and split **once**; V1 and V2 share the partition.
  If multiple images come from the same bird/sampling session, split by that **group id**.
- **Roboflow V3/V4:** split by **source clip/sequence**, never random frame (consecutive frames
  are near-duplicates).
- **Audio A1:** split by **recording id**, never by 4 s window.
- **Fusion / onset (C1):** **group holdout by flock** (already in `data/datamodule.split_by_flock`)
  — prevents the temporal/flock leakage that inflates most IoT-livestock papers.

### 6.2 Class imbalance

Newcastle (images) and Unhealthy/Noise (audio) are minorities → class-weighted or focal loss,
optional resampling, and **PR-AUC / macro-F1** for model selection (not accuracy). Report
per-class, not just aggregate.

### 6.3 Calibration, uncertainty, OOD

- **Calibration:** temperature scaling; report **ECE** (operators/vets act on the probability).
- **Uncertainty + OOD:** evidential heads or MC-dropout so models **abstain/flag** on
  out-of-distribution inputs (new farm, sensor fault, unseen lighting) — critical for vet trust
  and required by the roadmap (§2.1). Every deployed model publishes a confidence alongside its
  score.

### 6.4 MLOps spine (shared by all models)

`pip install -e ".[dev,track]"` · **MLflow** tracking + model registry on the VPS · config-driven
+ seeded runs · CI = `pytest` + `ruff` + a tiny smoke-train · **ONNX + onnxruntime** edge export
(`inference/export_edge.py`) · scores published under `ai/{owner}/{saloon}/...` (extend the topic
standard first — spec before code). Data path: InfluxDB → periodic parquet **training-store**
export matching `data_dictionary.md`.

### 6.5 Reproducibility & reporting

Mean ± std over **≥5 seeds**; paired **Wilcoxon** for lead-time / control vs. baseline; 95% CIs
via flock-level bootstrap; pinned deps; released code + configs.

---

## 7. Convergence: how the encoders become the thesis centrepiece

V3/V4 → **vision behavioural indices** → the fusion model's vision MLP. A1/A2 → the **audio
encoder**. Env → TCN/PatchTST encoder. Once longitudinal aligned data exists (real P0 or twin),
the existing `poultryai/models/multimodal.py` **gated cross-attention + modality-dropout** fusion
combines them into the **onset model**, evaluated by **lead-time @ fixed FPR** (`eval/leadtime.py`).
The control pillar then closes the loop: perception risk → controller action. Nothing here changes
the existing package's architecture — the component models produce the encoders it already expects.

**Disease coverage across modalities (the clean story):**

| Disease (of five) | Image (V1/V2) | Audio (A1) | Env/AQ + behaviour (V3/control) |
|---|---|---|---|
| Coccidiosis | ✅ direct | — | litter/AQ + activity |
| Newcastle | ✅ direct | respiratory phase | activity/distress |
| Infectious Bronchitis | — | ✅ respiratory | NH₃/CO₂ coupling |
| Avian Influenza | — | ✅ respiratory | activity collapse |
| Colibacillosis | — (Salmonella is a proxy-adjacent enteric extra, not equal) | partial | AQ/litter |

Images alone cover 2/5; audio + env/behaviour are precisely what close the respiratory gap — the
scientific case for multimodality.

---

## 8. Thesis-chapter mapping

| Model / result | Chapter |
|---|---|
| Dataset audit, leakage protocol, coverage analysis | System & Data |
| V1–V4 vision, A1–A2 audio (methods + ablations) | Methods (Detection/Perception) |
| Fusion onset model + lead-time results | Methods (Detection) + Results |
| C-1/C-2/C-3 control (twin + A/B) | Methods (Control) + Results |
| Calibration/OOD/robustness ablations | Results + Discussion |
| Safety shield, data governance | Ethics |

---

## 9. Phased milestone plan

| Phase | Models | Data | Status | Output / thesis artifact |
|---|---|---|---|---|
| **M1 — Spine + Vision-disease** | V1 (+V2) | Kaggle/Zenodo fecal | 🟢 **built** (gate verified; awaits GPU train) | fecal classifier+segmenter code, leakage-safe manifests, ONNX path, `ai/` fecal topic |
| **M2 — Audio** | A1, then A2 (SSL) | Mendeley | 🟢 **built** (gate + masker verified) | audio encoder + SSL pretrainer + label-efficiency script |
| **M3 — Vision-behaviour** | V3, V4 | Roboflow ×4 | 🟢 **built + runs** (tracker/indices on real frames) | detector→tracker→behavioural indices; posture health flag |
| **M4 — Control** | C-1 MPC, C-2 ES/RL, C-3 A/B | twin | 🟢 **built + runs** (A/B beats baseline) | closed-loop env control, twin A/B, safety shield |
| **M4b — Twin calibration (C-0)** | setpoint table + physics fit | real logs | 🔴 **pending P0 logs** | calibrated twin; absolute energy/incidence numbers |
| **M5 — Fusion (C1/C2)** | multimodal onset | twin-synth ✅ / real ❌ | 🟡 **pipeline built & verified on synthetic**; real data pending | lead-time @ fixed FPR + ablations — the early-warning claim |
| **M6 — Prove C3** | disease-aware control | twin → on-farm | 🔴 **pending on-farm** | A/B: incidence/mortality/FCR/energy — headline result |

M1–M4 and the M5 **pipeline** are **built and runnable now**. The remaining gates are not code:
M4b needs the cited setpoint table + real InfluxDB logs to calibrate the twin; M5 needs real
onset-labelled windows to turn the working pipeline into a result; M6 is the on-farm trial.

---

## 10. Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| Kaggle/Zenodo image overlap → inflated scores | invalid V1/V2 results | hash-dedup + shared, group-aware split (§6.1) |
| Only 2/5 diseases in images | overclaiming | frame fecal work as component demonstrator; close gap via audio + fusion |
| Roboflow frames are consecutive → leakage | inflated detection | split by clip/sequence (§6.1) |
| Small vision/audio sets | weak from-scratch models | transfer learning + SSL (A2); report as fine-tuned, validate index stability |
| No onset-labeled multimodal data | C1 blocked | start P0 logging now; twin-synthetic; SSL; sequence C1 last |
| Twin constants illustrative | unconvincing control | calibrate to real logs (C-0); label twin results as simulation until then |
| NH₃/CO₂ sensor drift | degrades control + labels | scheduled recalibration + software drift compensation |
| On-farm control safety | animal welfare / trust | twin-first + hard safety shield never overridden |

---

## 11. Immediate next actions

The build phase (V1–V4, A1–A2, C-1/C-2/C-3, C1 pipeline) is **complete and tested**. The frontier
is now GPU training + the data/hardware gates that turn working pipelines into results:

1. **GPU training pass** on the workstation: run each task's `train_*.py` (fecal, fecal_seg,
   vision detector, health_seg, audio; then `pretrain_audio_ssl.py` → A1 fine-tune →
   `eval_label_efficiency.py`). Log to **MLflow**; export **ONNX** for the RPi4/VPS.
2. **Fill the cited setpoint table** (Aviagen handbook) — unblocks C-0 twin calibration and the
   fusion labelling horizon.
3. **Start P0 farm logging** (secured InfluxDB) so real onset-labelled windows accumulate — the
   single gate that turns the M5 fusion pipeline and M6 control A/B into scientific results.
4. **Calibrate `saloontwin`** physics against the first real logs (M4b); then re-run the control
   A/B for quotable absolute numbers.
5. **Extend the MQTT topic standard** with the `ai/{owner}/{saloon}/...` root (spec before code)
   and stand up the ONNX **ML-serving** container on the VPS.
6. **Swap synthetic → real** in `scripts/train_fusion.py:load_windows`-equivalent (the generator
   is the only synthetic dependency) and report lead-time + ablations on real data.

*Component encoders first (real data, real deployment), fusion + control last (the headline
claims) — de-risked by SSL pretraining and the digital twin throughout.*
