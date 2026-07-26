"""V4 — Healthy/Sick posture segmentation (a disease-agnostic vision health cue).

Segments the bird and labels its apparent health (Healthy vs Sick) from the Roboflow
``roboflow_healthy_sick`` YOLOv8-seg dataset (505 images, one polygon each, ~balanced:
Healthy 241 / Sick 250, plus 14 background images). Complements the enteric signal (V1/V2
fecal) and the acoustic signal (A1) with an **appearance/posture** signal, and contributes a
``sick``/``distress`` cue to V3's behavioural indices and the fusion model.

Reuses infrastructure already built:
* the **clip-aware, leakage-free split** from V3 (``poultryai/tasks/vision/manifests/
  roboflow_healthy_sick.csv``) — Roboflow's shipped split leaks 11/75 source clips, so V4 uses
  the regrouped split;
* the U-Net / DeepLabV3 backbones and ONNX export from V2 (``poultryai.tasks.fecal_seg.model``).

Task: semantic segmentation over pixel classes ``{background, healthy, sick}``.

Modules
-------
* ``masks`` — YOLOv8-seg polygon .txt → semantic mask (torch-free; PIL only).
* ``data``  — reads the V3 manifest, serves (image, mask) pairs (needs torch/torchvision).
* ``eval``  — per-class IoU/Dice, mIoU, pixel-acc + derived image-level health report (numpy).
"""
from __future__ import annotations

# pixel classes: 0 background, then the two YOLO classes in dataset order (0=Healthy, 1=Sick)
SEG_CLASSES: tuple[str, ...] = ("background", "healthy", "sick")
SEG_CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(SEG_CLASSES)}
N_SEG_CLASSES: int = len(SEG_CLASSES)

# YOLO class id (from data.yaml: 0=Healthy, 1=Sick) → SEG pixel index
YOLO_TO_SEG: dict[int, int] = {0: SEG_CLASS_TO_IDX["healthy"], 1: SEG_CLASS_TO_IDX["sick"]}
