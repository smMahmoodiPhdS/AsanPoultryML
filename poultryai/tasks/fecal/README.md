# V1 — Fecal-image disease classifier

The first component demonstrator and the MLOps spine (script → train → test → ONNX
deploy). Classifies a fecal/dropping image into **Coccidiosis · Healthy · Newcastle ·
Salmonella**, covering 2 of the five thesis diseases (Coccidiosis, Newcastle) plus
Salmonella. See [`../../../docs/ai-model-plan.md`](../../../docs/ai-model-plan.md) (V1) for
the scientific framing. This is **not** the thesis' early-warning lead-time claim — that is
the multimodal onset model in `poultryai/models/multimodal.py`.

## Layout
```
poultryai/tasks/fecal/
  __init__.py     class list + filename→class maps
  splits.py       md5 dedup + stratified group split  (torch-free; the leakage gate)
  data.py         manifest-driven Dataset + transforms (torch/torchvision)
  model.py        transfer backbone + ONNX export      (torch; timm optional)
  eval.py         per-class P/R/F1, macro-F1, AUROC, confusion matrix, ECE (numpy/sklearn)
  manifest.csv    the frozen, leakage-safe split (committed for provenance)
scripts/
  build_fecal_manifest.py   build manifest.csv (run once)
  train_fecal.py            train + test + ONNX export
  eval_fecal.py             re-evaluate a checkpoint
configs/fecal.yaml
tests/test_fecal_splits.py  split-integrity unit tests (numpy-only)
```

## 1. Build the leakage-safe manifest (run once)
```bash
python scripts/build_fecal_manifest.py \
  --images Data/downloads/Kaggle/kaggle_chicken_disease/Train \
  --out    poultryai/tasks/fecal/manifest.csv
```
What it does, and **why it matters** (this is the scientific integrity step):

- **Exact dedup** by MD5. The corpus contains 277 byte-identical duplicate images; if they
  straddled train/test the test score would be inflated. They are collapsed into one group.
- **Label-conflict drop.** 3 sets of md5-identical images are filed under *two* different
  disease labels (e.g. `pcrcocci.228.jpg` == `pcrhealthy.83.jpg`); their true label is
  unknowable, so all 6 are dropped and reported.
- **Perceptual hash (pHash) is stored but NOT used to force groups by default.** On this
  corpus no perceptual threshold is stable — within a disease class the photos are so
  similar that even Hamming-radius 1 collapses hundreds of distinct images into one cluster.
  So `--max-hamming` defaults to `0` (exact dedup only); the pHash column is kept as a
  *manual-review flag* for suspected near-duplicates.
- **Stratified group split** 70/15/15, whole groups kept together, class proportions
  preserved (verified: Coccidiosis 0.31, Healthy 0.30, Newcastle 0.07, Salmonella 0.32 in
  every split).

Current result: `train 5662 / val 1203 / test 1196`, leakage check passes.

## 2. Train
```bash
pip install -e ".[dev,track]" torchvision timm     # torch stack (needs network)
python scripts/train_fecal.py --config configs/fecal.yaml
python scripts/train_fecal.py --config configs/fecal.yaml --smoke   # 1-epoch CPU pipeline check
```
EfficientNet-B0 (default) or ResNet-18 via torchvision; ConvNeXt/MobileViT via `timm` for the
accuracy/size ablation. Class-weighted cross-entropy (Newcastle minority → weight ≈ 2.4),
AdamW + cosine, early-stop on **val macro-F1**, optional MLflow tracking.

## 3. Test / report
Training ends with a report on the frozen test split **and** the PCR-confirmed slice
(higher-confidence ground truth). Regenerate from a checkpoint:
```bash
python scripts/eval_fecal.py --config configs/fecal.yaml --ckpt artifacts/fecal/best.pt
```
Metrics (per `docs/metrics.md`): per-class precision/recall/F1, macro-F1, macro AUROC,
confusion matrix, **ECE** (calibration). Reports are written to `artifacts/fecal/*.json`.

## 4. Deploy (edge)
`train_fecal.py` exports `artifacts/fecal_effb0.onnx`. Run with onnxruntime on the RPi4/VPS
and publish to MQTT:
```
ai/{owner}/{saloon}/fecal/class
ai/{owner}/{saloon}/fecal/score
```
(Add this `ai/` topic root to `Docs/Standards/mqtt-topic-structure.md` — spec before code.)

## Tests
```bash
python -m pytest tests/test_fecal_splits.py -q     # or run the functions directly if pytest absent
```
Guards the core invariant: no image (exact or grouped duplicate) may appear in more than
one split, and near-duplicates never merge across classes.

## Honest scope
A real, deployable classifier for 2-of-5 diseases + Salmonella, and the reusable
train→test→ONNX pipeline every later model inherits. The respiratory diseases (Infectious
Bronchitis, Avian Influenza) and the early-warning lead-time claim are addressed by the
audio model and the multimodal onset model, not here.
