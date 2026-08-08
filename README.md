# poultryai — Multimodal Early-Warning Models

The AI core of the thesis (Rasht Science and Research Branch of Iran Open
University). Predicts the onset of five broiler
diseases from **environmental time series + audio + vision** and evaluates *how early*
it warns versus a human supervisor. Designed to train on a workstation/GPU and deploy to
the **Raspberry Pi 4 edge** via ONNX.

> This is the scientific centre of the project. See `docs/methodology.md` for the design
> rationale and literature grounding, and `docs/metrics.md` for the lead-time protocol.

## Layout
```
poultryai/
  data/        schema (Window, Disease), datamodule (flock-grouped splits), collate
  features/    audio log-mel front-end (SpecAugment); env feature helpers
  models/      env encoder (TCN | PatchTST), audio CNN, vision MLP, attention fusion,
               full MultimodalEarlyWarning + LightningModule
  train/       focal loss, metrics (PR-AUC, macro-F1, ECE)
  eval/        lead-time evaluation + FPR-constrained threshold tuning
  inference/   ONNX/TorchScript export for RPi4; streaming MQTT edge inference
  utils/       seeding, config
configs/       default.yaml (config-driven experiments)
docs/          methodology, metrics, data dictionary
scripts/       train.py entry point
tests/         unit tests (lead-time logic)
```

## Quickstart
```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,track]"          # add ".[edge]" on the RPi4 (onnxruntime+paho)
pytest -q                               # runs the lead-time unit tests (no GPU needed)
# implement scripts/train.py:load_windows for your data, then:
python scripts/train.py --config configs/default.yaml
```

## Design highlights (for review)
- **Onset-aware, lead-time-first evaluation** — answers "how much earlier than a human?"
  rather than only window accuracy.
- **Flock-grouped splitting** — prevents the temporal leakage that inflates results in
  much IoT-livestock work.
- **Modality dropout + attention fusion** — graceful degradation when a sensor drops out.
- **Edge-exportable** — log-mel front-end is inside the model, so ONNX takes raw audio.
- **Modern, cited backbones** — PatchTST/Medformer-style time-series, AST/CNN audio,
  focal loss, temperature-scaling calibration.

## Status
Reference implementation and protocol are complete and importable. The real dataset
loader and trained weights follow data collection (Phase P2 in the thesis plan).
