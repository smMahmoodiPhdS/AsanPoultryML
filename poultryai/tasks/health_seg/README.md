# V4 — Healthy/Sick posture segmentation

A **disease-agnostic** vision health cue: segment the bird and label its apparent health
(Healthy vs Sick). Complements the enteric signal (V1/V2 fecal) and the acoustic signal (A1)
with an appearance/posture signal, and feeds V3's `distress_index` and the fusion model. See
[`../../../docs/ai-model-plan.md`](../../../docs/ai-model-plan.md) (V4).

Task: semantic segmentation over pixel classes `{background, healthy, sick}`.

## Data
`PhdThesis/Data/downloads/Roboflow/roboflow_healthy_sick` — YOLOv8-seg polygons, **505 images**, one
polygon each, **well balanced** (Healthy 241 / Sick 250) with 14 background images, exported at
416×416.

## Reuse (no new split, no new manifest)
V4 consumes V3's committed **clip-aware, leakage-free** split
`poultryai/tasks/vision/manifests/roboflow_healthy_sick.csv` (Roboflow's shipped split leaks
11/75 source clips; the regrouped split is `train 424 / val 39 / test 42`). It also reuses V2's
U-Net / DeepLabV3 backbones and ONNX export (`poultryai.tasks.fecal_seg.model`) with 3 classes.

## Layout
```
poultryai/tasks/health_seg/
  __init__.py   SEG classes (background/healthy/sick), YOLO→SEG map
  masks.py      YOLOv8-seg polygon .txt → semantic mask     (torch-free)
  data.py       reads the V3 manifest → (image, mask) pairs (torch/torchvision)
  eval.py       IoU/Dice/mIoU + image-level health report   (numpy)
scripts/train_health_seg.py
configs/health_seg.yaml
tests/test_health_seg.py
```

## Run
```bash
# 1) ensure the clip-aware manifest exists (built by V3):
python scripts/build_roboflow_manifest.py \
  --root PhdThesis/Data/downloads/Roboflow/Data/downloads --out-dir poultryai/tasks/vision/manifests

# 2) train (needs torch stack):
pip install -e ".[dev,track]" torchvision
python scripts/train_health_seg.py --config configs/health_seg.yaml
python scripts/train_health_seg.py --config configs/health_seg.yaml --smoke   # CPU pipeline check
```

## Report
Per-class **IoU/Dice**, **mIoU** (+ foreground-only), pixel accuracy, and a derived
**image-level health report** (dominant non-background class → Healthy/Sick accuracy + per-class
recall) so V4 is comparable to a Healthy/Sick classifier. Written to
`artifacts/health_seg/test_report.json`.

## Scope & how it feeds the thesis
Small dataset ⇒ transfer-learned, framed as a **component demonstrator**. It provides a second,
spatially-grounded health signal — the `sick` cue V3's `distress_index` uses and an independent
vision input for the multimodal fusion model — deployed on the edge via ONNX and published
under `ai/{owner}/{saloon}/health/...`.
