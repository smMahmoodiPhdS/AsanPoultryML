"""YOLOv8-seg polygon labels → semantic mask (torch-free, PIL + numpy only).

Each label file is one or more lines ``<class> x1 y1 x2 y2 ... xn yn`` with **normalised**
polygon coordinates (0..1). Empty/missing files mean a background-only image (14 of the 505).
We rasterise every polygon into an (H, W) uint8 mask of SEG class indices (0 = background).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import N_SEG_CLASSES, YOLO_TO_SEG


def parse_yoloseg_label(label_path: str | Path) -> list[tuple[int, np.ndarray]]:
    """Return list of (yolo_class_id, polygon Nx2 normalised) for a label file."""
    p = Path(label_path)
    out: list[tuple[int, np.ndarray]] = []
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:                       # class + >=3 (x,y) pairs
            continue
        cid = int(float(parts[0]))
        coords = np.array(parts[1:], dtype=np.float64)
        coords = coords[: (len(coords) // 2) * 2].reshape(-1, 2)
        out.append((cid, coords))
    return out


def rasterise_yoloseg(polys: list[tuple[int, np.ndarray]], height: int, width: int) -> np.ndarray:
    """Rasterise normalised YOLO-seg polygons into an (H, W) uint8 SEG-index mask."""
    from PIL import Image, ImageDraw
    mask = Image.new("L", (width, height), 0)    # 0 = background
    draw = ImageDraw.Draw(mask)
    for cid, poly in polys:
        seg_idx = YOLO_TO_SEG.get(cid)
        if seg_idx is None or len(poly) < 3:
            continue
        pts = [(float(x) * width, float(y) * height) for x, y in poly]
        draw.polygon(pts, fill=seg_idx)
    return np.asarray(mask, dtype=np.uint8)


def load_example(image_path: str | Path, label_path: str | Path):
    """Return (PIL RGB image, uint8 mask HxW) for one (image, label) pair."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    polys = parse_yoloseg_label(label_path)
    mask = rasterise_yoloseg(polys, H, W)
    return img, mask


def label_pixel_hist(label_path: str | Path, height: int, width: int) -> np.ndarray:
    """Class-pixel counts for one mask (for pixel-weight estimation without keeping images)."""
    polys = parse_yoloseg_label(label_path)
    mask = rasterise_yoloseg(polys, height, width)
    return np.bincount(mask.reshape(-1), minlength=N_SEG_CLASSES)[:N_SEG_CLASSES]
