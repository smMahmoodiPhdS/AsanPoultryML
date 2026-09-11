# V2 — Fecal lesion/region segmentation

Turns V1's disease *classifier* into a *segmenter*: it localises the dropping/lesion region
**and** labels its disease, so every prediction carries a spatial "why" — the explainability
the roadmap (§2.1) and the vet-trust argument require. See
[`../../../docs/ai-model-plan.md`](../../../docs/ai-model-plan.md) (V2).

Task: semantic segmentation over pixel classes
`{background, coccidiosis, healthy, newcastle, salmonella}` (V1's four diseases + background).

## The key property: it reuses V1's leakage-safe split

Data is the **Zenodo 4628934** LabelMe JSONs (one polygon per image, image embedded as
base64) — the **same source corpus** as V1's Kaggle images. Each JSON's `imagePath` carries
the identical filename (`cocci.1819.jpg`, `ncd.24.jpg`, …), so V2 **joins by filename to V1's
`manifest.csv` and copies its train/val/test assignment**. This guarantees no image V1
trained on is ever tested by V2 (and vice-versa) — the cross-model leakage guard from
`ai-model-plan.md` §6.1.

Verified on the built manifests (`tests/test_fecal_seg.py::test_split_inheritance_no_leakage`):

```
seg-test ∩ V1-train = 0      seg-train ∩ V1-test = 0      (held-out sets shared: 1011)
matched to V1: 6811 / 6812   (1 image has no V1 twin → stratified fallback)
inherited split: train 4789 / val 1012 / test 1011
```

## Layout
```
poultryai/tasks/fecal_seg/
  __init__.py   SEG classes (background + V1 diseases), label maps
  masks.py      LabelMe head-read + base64 decode + polygon→mask raster  (torch-free)
  data.py       manifest-driven seg Dataset + paired aug                 (torch/torchvision)
  model.py      U-Net (edge) + torchvision DeepLabV3 (server) + ONNX     (torch)
  eval.py       per-class IoU/Dice, mIoU, pixel-acc + image-level report (numpy)
  manifest.csv  frozen, inherits V1's split (committed for provenance)
scripts/
  build_fecal_seg_manifest.py   join Zenodo→V1 split (run once)
  train_fecal_seg.py            train + test + ONNX
configs/fecal_seg.yaml
tests/test_fecal_seg.py         mask + metrics + split-inheritance tests (numpy-only)
```

## 1. Build the manifest (run once)
```bash
python scripts/build_fecal_seg_manifest.py \
  --zenodo "PhdThesis/Data/downloads/zenodo_4628934_poultry_fecal/content (2)/imgSegmentation" \
  --v1-manifest poultryai/tasks/fecal/manifest.csv \
  --out poultryai/tasks/fecal_seg/manifest.csv
```
Reads only JSON *headers* (not the embedded ~MB images), so it runs over 6 812 files in ~6 s.

## 2. Train
```bash
pip install -e ".[dev,track]" torchvision                 # torch stack (needs network)
python scripts/train_fecal_seg.py --config configs/fecal_seg.yaml
python scripts/train_fecal_seg.py --config configs/fecal_seg.yaml --smoke   # CPU pipeline check
```
Default **U-Net** (compact, ONNX-friendly edge model); switch `model.backbone:
deeplabv3_resnet50` for the higher-capacity server comparison. Loss = pixel-weighted
cross-entropy + soft Dice (background dominates → the weights up-weight the rare disease
pixels). Selects on **foreground mIoU**.

## 3. Report
Test-time metrics: per-class **IoU/Dice**, **mIoU** (and foreground-only mIoU), pixel
accuracy, plus a **derived image-level disease report** (dominant non-background class) so V2
is directly comparable to V1's macro-F1/confusion matrix. Written to
`artifacts/fecal_seg/test_report.json`.

## 4. Deploy
`train_fecal_seg.py` exports `artifacts/fecal_seg_unet.onnx`. On the edge, the mask both
drives the disease label and provides the localisation overlay for the operator/vet UI;
publish alongside V1 under `ai/{owner}/{saloon}/fecal/...`.

## How it feeds the thesis
V2 is the **explainability layer** over V1 (localised lesion + class), a mask source for
V1's mask-guided augmentation, and a second, spatially-grounded disease signal for the vision
modality. Same honest scope as V1: 2-of-5 diseases + Salmonella; the respiratory diseases and
the early-warning lead-time claim are the audio model and the multimodal onset model.
