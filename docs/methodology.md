# AI Methodology — Multimodal Early Warning for Broiler Disease

This document states the scientific design of the AI core of the thesis: the problem
formulation, modality choices, model architecture, training regime, evaluation, and how
each choice is grounded in current literature. It is written to be defensible at a PhD
level and to be reproduced from the code in `poultryai/`.

## 1. Problem formulation

We frame early disease detection as **multi-label, onset-aware temporal classification**.
For each saloon we form time-aligned windows of three modalities and predict, per window,
a probability for each of the five target diseases. Crucially we retain the
veterinary-confirmed **onset time** per flock so the objective and evaluation reward
*early* detection, not merely post-hoc window accuracy. This distinguishes the work from
systems that report frame/window accuracy without a lead-time analysis.

Let a flock produce windows \(x_t\) with scores \(s_t = f_\theta(x_t)\in[0,1]^5\). The
operational question (RQ2) is: at a controlled false-alarm rate, how many hours before
onset does \(s_t\) first cross an alarm threshold, versus a human supervisor?

## 2. Modalities and why

| Modality | Signal | Rationale |
|---|---|---|
| Environmental time series | temp, RH, light, NH₃, CO₂ (1/min) | Slow physiological/AQ drivers; NH₃/CO₂ precede and aggravate respiratory disease. |
| **Audio** | I²S MEMS mic, log-mel spectrogram | Respiratory diseases (esp. Infectious Bronchitis) produce coughs/rales audible before visible decline. Acoustic detection matches/*exceeds* human experts in recent farm studies. |
| Vision | camera-derived behavioural indices | Activity, distribution/piling, feeding visits, distress — behavioural collapse is an early, disease-agnostic cue. |
| Flock metadata | age, density, mortality prior | Conditions the prediction; disease base-rates vary strongly with age. |

Audio is a first-class modality here specifically to sharpen respiratory-disease
detection, consistent with 2024–2026 work showing spectrogram-based transformers/CNNs
detecting chicken cough in real farm noise.

## 3. Architecture

Intermediate (feature-level) fusion of three modality encoders (`poultryai/models/`):

* **Environmental encoder** — selectable **TCN** (Bai et al., 2018) for cheap causal
  streaming on the edge, or a **PatchTST**-style channel-independent patch Transformer
  (Nie et al., 2023) for long context. Patch Transformers are current SOTA-competitive on
  multivariate time series and, per Medformer (Zhang et al., NeurIPS 2024), multi-
  granularity patching is effective for *medical* time-series classification — a close
  analogue to physiological monitoring.
* **Audio encoder** — log-mel front-end (`features/audio.py`) with SpecAugment, feeding a
  compact depthwise-separable CNN for the edge model; an Audio Spectrogram Transformer
  (Gong et al., 2021) can be substituted for the server model. Log-mel + spectrogram
  transformer/CNN is the dominant, validated recipe in poultry respiratory acoustics
  (SmartEars 2025; ASCT-CC 2026).
* **Vision encoder** — an MLP over pre-extracted behavioural indices (kept modular so the
  upstream detector/tracker — e.g. a YOLO-family model — can evolve independently).
* **Fusion** — gated cross-attention with a learnable query over modality tokens plus
  **modality dropout**, so the model degrades gracefully when a sensor drops out (routine
  on-farm). This is the robustness analogue of multimodal EHR models that handle missing/
  irregular modalities (MSTFT, 2024).

Per-disease linear heads keep calibration independent across diseases.

## 4. Training

* **Loss** — focal BCE (Lin et al., 2017) with per-disease `pos_weight` to counter the
  severe imbalance of onset windows. Label smoothing optional.
* **Optimisation** — AdamW + cosine schedule, mixed precision, early stopping on
  validation macro average-precision.
* **Splitting** — **group holdout by flock** (`data/datamodule.split_by_flock`) to prevent
  temporal/flock leakage — the single most common way IoT-livestock papers overstate
  performance.
* **Calibration** — post-hoc temperature scaling (Guo et al., 2017); operators act on the
  probability, so calibration (ECE) is a reported first-class metric.
* **Reproducibility** — fixed seeds, pinned deps (`pyproject.toml`), config-driven runs,
  experiment tracking (MLflow/W&B).

## 5. Evaluation

Primary: **detection lead time** at a fixed flock-level false-alarm budget
(`eval/leadtime.py`, and `docs/metrics.md`). Secondary: per-disease PR-AUC (rare-positive
appropriate), macro-F1, and ECE. Baselines: (i) a human-supervisor timestamp, (ii)
threshold-on-single-sensor rules (e.g. NH₃ limit), (iii) unimodal ablations to quantify
the marginal value of audio and vision.

## 6. Edge deployment

The whole model — including the log-mel front-end — exports to ONNX/TorchScript
(`inference/export_edge.py`) and runs on the RPi4 with onnxruntime, consuming raw
waveforms so no preprocessing is re-implemented on the edge. Streaming inference publishes
scores back to MQTT under an `ai/` topic root (`inference/realtime.py`).

## 7. Ethics and data governance

Flock data is owner-namespaced and access-controlled. The model outputs early warnings,
not diagnoses; veterinary confirmation remains the ground-truth and the actuator safety
interlocks are never overridden by the AI.

## Selected references
- Nie et al. (2023), *A Time Series is Worth 64 Words: Long-term Forecasting with
  Transformers* (PatchTST).
- Zhang et al. (2024, NeurIPS), *Medformer: Multi-Granularity Patching Transformer for
  Medical Time-Series Classification*.
- Bai, Kolter, Koltun (2018), *An Empirical Evaluation of Generic Convolutional and
  Recurrent Networks for Sequence Modeling* (TCN).
- Gong, Chung, Glass (2021), *AST: Audio Spectrogram Transformer*.
- Lin et al. (2017), *Focal Loss for Dense Object Detection*.
- Guo et al. (2017), *On Calibration of Modern Neural Networks*.
- SmartEars (2025) and ASCT-CC (2026): spectrogram-based chicken cough detection in farms.

*(Full bibliographic details to be finalised in the thesis reference manager.)*
