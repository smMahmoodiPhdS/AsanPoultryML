"""V1 — Fecal-image disease classifier (component demonstrator).

Covers Coccidiosis + Newcastle of the five target diseases, plus Salmonella and Healthy,
from the public poultry fecal-image corpus (Kaggle images / Zenodo 4628934 masks).

This is a *component demonstrator* and the MLOps spine (script -> train -> test ->
ONNX deploy), **not** the thesis' early-warning lead-time claim (that is the multimodal
onset model in ``poultryai.models.multimodal``). See ``docs/ai-model-plan.md`` (V1).

Modules
-------
* ``splits``  — content-hash dedup + stratified, group-aware split (torch-free; the
  leakage gate that makes the result valid). Unit-tested without torch.
* ``data``    — manifest-driven ``Dataset`` + transforms (needs torch/torchvision).
* ``model``   — transfer-learning backbone with a 4-logit head (needs torch).
* ``eval``    — per-class P/R/F1, macro-F1, AUROC, confusion matrix, ECE (needs sklearn).
"""

CLASSES: tuple[str, ...] = ("coccidiosis", "healthy", "newcastle", "salmonella")
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

# Map the raw Kaggle filename prefixes (and CSV label strings) to canonical class names.
PREFIX_TO_CLASS: dict[str, str] = {
    "cocci": "coccidiosis", "pcrcocci": "coccidiosis",
    "healthy": "healthy", "pcrhealthy": "healthy",
    "ncd": "newcastle", "pcrncd": "newcastle",
    "salmo": "salmonella", "pcrsalmo": "salmonella",
}
CSV_LABEL_TO_CLASS: dict[str, str] = {
    "coccidiosis": "coccidiosis",
    "healthy": "healthy",
    "new castle disease": "newcastle",
    "salmonella": "salmonella",
}
