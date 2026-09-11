# A1 — Acoustic health / abnormality classifier (the audio encoder)

Trains on the Mendeley chicken-vocalization corpus to produce the **audio encoder** the
multimodal fusion model reuses. This is the modality that covers the **respiratory-disease
gap** (Infectious Bronchitis, Avian Influenza, the respiratory phase of Newcastle) that the
image models cannot see — cough / rale / sneeze are audible before visible decline. See
[`../../../docs/ai-model-plan.md`](../../../docs/ai-model-plan.md) (A1).

## Data
`PhdThesis/Data/downloads/mendeley_vocalization/Chicken_Audio_Dataset` — **346 recordings**, 48 kHz mono:
Healthy 139 / Unhealthy 121 / Noise 86, durations 0.5 s – 906 s.

Honest scope: the labels are a **binary health signal** (Healthy vs Unhealthy) + an explicit
**Noise** (farm background) class — so A1 is *acoustic health detection*, not per-disease
acoustic diagnosis. Noise doubles as a vocalization-vs-background gate.

## Leakage-safe, balance-aware pipeline
- **Recording-level split** (stratified by class): windows from one recording never cross
  train/val/test. Built: `train 242 / val 52 / test 52` files.
- **Windowing** (`windowing.py`): long clips → fixed 4 s windows (train hop 2 s = 50 % overlap);
  clips < 4 s are zero-padded; a **per-recording cap** (20) stops the 906 s Healthy clip from
  dominating. Effective train windows are balanced (~977 / 859 / 842).
- **Recording-level metrics** are the reported unit (a farmer cares about the bird, not a 4 s
  window): window probabilities are averaged per recording before scoring.

## Reuse (encoder lifts straight into the fusion model)
```
raw waveform → LogMelFrontend (features.audio) → AudioCNNEncoder (models.audio_encoder) → Linear(3)
```
The log-mel front-end is **inside** the model, so ONNX export takes the raw waveform — no
preprocessing is re-implemented on the RPi4 edge. `model.embed()` is exactly the fusion model's
audio branch. Swap `encoder: ast` for the server-side Audio Spectrogram Transformer.

## Layout
```
poultryai/tasks/audio/
  __init__.py   classes + folder→class map
  windowing.py  clip → window offsets, pad/trim, per-file cap   (torch-free)
  splits.py     scan wavs (wave stdlib) + recording-level split (torch-free)
  data.py       load + resample 48k→16k + window                (torch/torchaudio)
  model.py      front-end + encoder + head; ONNX from raw wav   (torch)
  eval.py       window- & recording-level macro-F1/AUROC/ECE    (numpy/sklearn)
  manifest.csv  committed recording-level split
scripts/build_audio_manifest.py · scripts/train_audio.py
configs/audio.yaml · tests/test_audio.py
```

## Run
```bash
python scripts/build_audio_manifest.py \
  --root PhdThesis/Data/downloads/mendeley_vocalization/Chicken_Audio_Dataset \
  --out poultryai/tasks/audio/manifest.csv          # torch-free; run once

pip install -e ".[dev,track,audio]" torchaudio
python scripts/train_audio.py --config configs/audio.yaml
python scripts/train_audio.py --config configs/audio.yaml --smoke   # CPU pipeline check
```

## Reports
Window-level and recording-level: per-class precision/recall/F1, macro-F1, macro AUROC, a
**healthy-vs-rest AUROC** (the clinically relevant collapse), ECE, confusion matrix — to
`artifacts/audio/test_{window,recording}.json`. Deploy: ONNX + onnxruntime; publish
`ai/{owner}/{saloon}/audio/{class,score}`.

## A2 — Self-supervised pretraining (masked-spectrogram / audio-MAE)

The label-efficiency contribution: pretrain **A1's exact encoder** on *unlabeled* audio, then
fine-tune A1 from it. Reuses the same `LogMelFrontend` + `AudioCNNEncoder`, so the pretrained
conv weights transfer 1:1.

- **`ssl.py`** — `MaskedSpecAutoencoder`: zero out ~60 % of the log-mel in 8×8 time-frequency
  **patches**, encode the corrupted spectrogram (`AudioCNNEncoder` stem+body), decode back, and
  take the reconstruction MSE **on the masked region** (the MAE signal). The patch masker is
  numpy-only and unit-tested (`tests/test_audio_ssl.py`).
- **No pretraining leakage**: SSL uses only the manifest's `train`+`val` recording windows —
  **never test**.

```bash
# 1) pretrain the encoder on unlabeled audio → artifacts/audio_ssl/encoder.pt
python scripts/pretrain_audio_ssl.py --config configs/audio_ssl.yaml

# 2) fine-tune A1 from it: set model.init_encoder in configs/audio.yaml, then
python scripts/train_audio.py --config configs/audio.yaml

# 3) the payoff — does SSL help when labels are scarce?
python scripts/eval_label_efficiency.py --config configs/audio.yaml \
    --ssl-encoder artifacts/audio_ssl/encoder.pt --fractions 0.1 0.25 0.5 1.0
```
`eval_label_efficiency.py` trains A1 at each label fraction twice (scratch vs SSL-init),
subsampling **recordings** (stratified, never touching val/test), and reports the macro-F1 gap —
expected to be largest at low label fractions (the low-resource-farm case). Needs torchaudio +
scikit-learn.
