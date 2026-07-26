# C1 — Multimodal onset model, wired end-to-end on twin-synthetic data

The thesis centrepiece — `poultryai/models/multimodal.py` (env+audio+vision → per-disease onset
risk) with the **lead-time** metric — needs time-aligned windows with veterinary onset times,
which only real longitudinal data or the twin provide. This package makes the whole
**train → lead-time evaluation → ablation** pipeline runnable **now**, on synthetic data, so it is
correct and connected *before* real data exists. See
[`../../../docs/ai-model-plan.md`](../../../docs/ai-model-plan.md) §7 and `docs/metrics.md`.

> **Simulation, not a result.** Env comes from `saloontwin`; per-disease **pre-onset signatures**
> are injected across modalities. Absolute lead times validate the *plumbing*; real numbers come
> from training on real / twin-calibrated data.

## What it generates (`synth.py`)
Valid `schema.Window` flocks: env `(240, 5)` from the twin (+ a light program for `lux`), a raw
`4 s @ 16 kHz` audio waveform, vision indices `(T_vis, 5)`, flock metadata, multi-label onset
labels, and the per-disease `onset_offset` the metric needs. Disease realism is modality-specific
so the ablations mean something:

| disease | audio (cough) | env (NH₃/CO₂) | vision (activity/feed/distress) |
|---|---|---|---|
| infectious bronchitis, avian influenza, newcastle (respiratory) | **strong** | drift | moderate |
| coccidiosis, colibacillosis (enteric) | weak | mild | **strong** |

A signature **ramps up over a pre-onset horizon** before the onset time; windows within a
labelling horizon are positive. Healthy flocks carry no labels and `onset_offset = +inf`.

## Pipeline (end-to-end, verified)
```
tasks.fusion.synth → schema.Window flocks
   → data.datamodule.split_by_flock         (flock-grouped: no temporal leakage)
   → MultimodalEarlyWarning + FocalBCE       (train; modality-dropout for C2 robustness)
   → tasks.fusion.leadtime_io.build_flock_series
   → eval.leadtime: tune threshold on val @ fixed flock-FPR, freeze, evaluate on test
```
The torch-free part (generation + lead-time protocol) runs now; on the signature-proxy oracle
score all diseases are detected before onset with 30–56 h median lead — proving the onset
labelling, flock split, I/O contract, and lead-time metric are correct and connected.

## Run
```bash
python scripts/build_synth_windows.py --n-flocks 16 --flock-days 4        # torch-free; provenance
pip install -e ".[dev,track]" torchaudio
python scripts/train_fusion.py --config configs/fusion.yaml               # train + lead-time
python scripts/train_fusion.py --config configs/fusion.yaml --smoke       # CPU pipeline check
python scripts/train_fusion.py --config configs/fusion.yaml --ablation audio_off   # C2 ablation
```
`--ablation {audio_off,vision_off,env_off}` zeroes a modality's presence mask to quantify its
marginal lead-time value — audio should matter most for the respiratory diseases (the scientific
point of the multimodal design).

## Layout
```
poultryai/tasks/fusion/
  synth.py        twin-synthetic flock/window generator + disease signatures  (torch-free)
  leadtime_io.py  windows + scores → FlockSeries; signature-proxy oracle       (torch-free)
scripts/ build_synth_windows.py · train_fusion.py
configs/fusion.yaml · tests/test_fusion_synth.py
```

## How it feeds the thesis (C1 → C2 → C3)
- **C1** — this lead-time pipeline; swap synthetic for real/twin-calibrated windows to get results.
- **C2** — `--ablation` + the model's `modality_dropout` give the graceful-degradation / sensor-loss
  story and the audio/vision marginal-value ablations.
- **C3** — the same disease-risk score feeds the control pillar's risk-aware ventilation coupling.
- The trained encoders from V1–V4 / A1–A2 initialise the env/audio/vision branches once real data
  is available (they already share the front-ends).
