"""LabelMe JSON → (image, semantic mask) — torch-free (PIL + numpy only).

Each Zenodo JSON embeds the source image as base64 (`imageData`) and one or more polygons
(`shapes[].points`) labeled with a disease. We decode the image and rasterise every polygon
into a single-channel mask whose pixel values are the SEG class index (0 = background).

Two entry points:
* :func:`read_header`  — cheap: extract only ``imagePath`` + first label from the file head
  (does *not* decode the ~MB base64 image). Used to build the manifest over 6 800+ files fast.
* :func:`load_example` — full: decode image + rasterise mask for training/eval.
"""
from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import SEG_CLASS_TO_IDX, SHAPE_LABEL_TO_CLASS

_RX_PATH = re.compile(rb'"imagePath"\s*:\s*"([^"]+)"')
_RX_LABEL = re.compile(rb'"label"\s*:\s*"([^"]+)"')


@dataclass(slots=True)
class SegHeader:
    json_path: str
    basename: str          # e.g. "cocci.1819.jpg" — the join key to V1's manifest
    disease: str           # canonical V1 class name
    shape_label: str       # raw LabelMe label


def read_header(json_path: str | Path, head_bytes: int = 65536) -> SegHeader | None:
    """Extract the join key + label from the file head without decoding the image.

    Returns None if the file has no imagePath/label in the head or an unmapped label.
    """
    with open(json_path, "rb") as fh:
        head = fh.read(head_bytes)
    mp, ml = _RX_PATH.search(head), _RX_LABEL.search(head)
    if not mp or not ml:
        return None
    raw_path = mp.group(1).decode("utf-8", "replace").replace("\\\\", "/").replace("\\", "/")
    basename = raw_path.rsplit("/", 1)[-1]
    shape_label = ml.group(1).decode("utf-8", "replace").strip().lower()
    disease = SHAPE_LABEL_TO_CLASS.get(shape_label)
    if disease is None:
        return None
    return SegHeader(str(json_path), basename, disease, shape_label)


def _decode_image(d: dict) -> "PILImage":
    from PIL import Image
    data = d.get("imageData")
    if data:
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    # fallback: imagePath on disk (rare — Zenodo embeds imageData)
    p = Path(d["imagePath"])
    return Image.open(p).convert("RGB")


def rasterise_mask(shapes: list[dict], height: int, width: int) -> np.ndarray:
    """Rasterise LabelMe polygons into an (H, W) uint8 mask of SEG class indices."""
    from PIL import Image, ImageDraw
    mask = Image.new("L", (width, height), 0)   # 0 = background
    draw = ImageDraw.Draw(mask)
    for sh in shapes:
        disease = SHAPE_LABEL_TO_CLASS.get(str(sh.get("label", "")).strip().lower())
        if disease is None:
            continue
        idx = SEG_CLASS_TO_IDX[disease]
        pts = [(float(x), float(y)) for x, y in sh.get("points", [])]
        if len(pts) >= 3:
            draw.polygon(pts, fill=idx)
        elif len(pts) == 2:                     # rectangle shape_type
            (x0, y0), (x1, y1) = pts
            draw.rectangle([x0, y0, x1, y1], fill=idx)
    return np.asarray(mask, dtype=np.uint8)


def load_example(json_path: str | Path):
    """Return (PIL RGB image, uint8 mask HxW) fully decoded — for training/eval."""
    with open(json_path) as f:
        d = json.load(f)
    img = _decode_image(d)
    w = int(d.get("imageWidth") or img.width)
    h = int(d.get("imageHeight") or img.height)
    if (img.width, img.height) != (w, h):       # trust the actual decoded image
        w, h = img.width, img.height
    mask = rasterise_mask(d.get("shapes", []), h, w)
    return img, mask
