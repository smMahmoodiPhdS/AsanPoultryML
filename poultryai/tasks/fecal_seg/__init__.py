"""V2 — Fecal lesion/region segmentation (component demonstrator, explainability layer).

Turns the V1 disease *classifier* into a *segmenter*: it localises the dropping/lesion
region **and** labels its disease, so a prediction comes with a spatial "why" — the
explainability the roadmap (§2.1) and vet-trust argument require. Also enables mask-guided
augmentation and a segmentation-guided classifier for V1.

Data: the Zenodo 4628934 LabelMe JSONs (one polygon per image, embedded base64 image),
which are the **same source corpus** as V1's Kaggle images. V2 therefore **inherits V1's
train/val/test split by filename** (``build_fecal_seg_manifest.py``) so no image V1 trained
on is ever tested by V2 — the cross-model leakage guard from ``docs/ai-model-plan.md`` (§6.1).

Task framing: semantic segmentation over pixel classes
``{background, coccidiosis, healthy, newcastle, salmonella}`` — segment the dropping and
classify its disease in one pass. Reuses V1's disease class names.

Modules
-------
* ``masks``  — base64 image decode + polygon→mask rasterisation (torch-free; PIL only).
* ``data``   — manifest-driven segmentation ``Dataset`` (needs torch/torchvision).
* ``model``  — lightweight U-Net (edge) + optional torchvision DeepLabV3 (server); ONNX.
* ``eval``   — per-class IoU/Dice, mean IoU, pixel-acc + a derived image-level report.
"""
from __future__ import annotations

from poultryai.tasks.fecal import CLASSES as DISEASE_CLASSES

# pixel-level classes: index 0 is background, then the four disease classes (V1 order)
SEG_CLASSES: tuple[str, ...] = ("background", *DISEASE_CLASSES)
SEG_CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(SEG_CLASSES)}
N_SEG_CLASSES: int = len(SEG_CLASSES)

# LabelMe shape labels (Zenodo) → V1 canonical disease names
SHAPE_LABEL_TO_CLASS: dict[str, str] = {
    "cocci": "coccidiosis", "healthy": "healthy",
    "ncd": "newcastle", "salmo": "salmonella",
}
