"""Torch-free tests for V2: mask rasterisation, seg metrics, and split inheritance.

Guards the two properties that make V2 valid: (1) polygons rasterise to correct class
masks, and (2) V2 inherits V1's split so no image V1 trained on is tested by V2.
"""
import csv
from pathlib import Path

import numpy as np

from poultryai.tasks.fecal_seg import N_SEG_CLASSES, SEG_CLASS_TO_IDX
from poultryai.tasks.fecal_seg.masks import rasterise_mask
from poultryai.tasks.fecal_seg.eval import (dominant_class, seg_metrics, update_confusion)


def test_rasterise_polygon_class():
    shapes = [{"label": "cocci", "points": [[2, 2], [8, 2], [8, 8], [2, 8]]}]
    mask = rasterise_mask(shapes, height=10, width=10)
    assert mask.shape == (10, 10)
    assert mask.max() == SEG_CLASS_TO_IDX["coccidiosis"]
    assert mask.min() == 0                       # background present
    assert (mask == SEG_CLASS_TO_IDX["coccidiosis"]).sum() > 0


def test_rasterise_unmapped_label_ignored():
    mask = rasterise_mask([{"label": "garbage", "points": [[0, 0], [5, 0], [5, 5]]}], 8, 8)
    assert mask.max() == 0                        # nothing drawn


def test_dominant_class_ignores_background():
    m = np.zeros((10, 10), dtype=np.uint8)
    m[:3, :3] = SEG_CLASS_TO_IDX["salmonella"]
    assert dominant_class(m) == SEG_CLASS_TO_IDX["salmonella"]
    assert dominant_class(np.zeros((4, 4), dtype=np.uint8)) == 0


def test_seg_metrics_perfect():
    conf = np.zeros((N_SEG_CLASSES, N_SEG_CLASSES), dtype=np.int64)
    pred = np.array([[0, 1], [2, 3]])
    update_confusion(conf, pred, pred)            # perfect prediction
    m = seg_metrics(conf)
    assert abs(m["pixel_accuracy"] - 1.0) < 1e-9
    for c in ("background", "coccidiosis"):
        assert abs(m["per_class"][c]["iou"] - 1.0) < 1e-9


def test_split_inheritance_no_leakage():
    """If the built manifests exist, assert the V1/V2 cross-model guarantee holds."""
    v1p = Path("poultryai/tasks/fecal/manifest.csv")
    v2p = Path("poultryai/tasks/fecal_seg/manifest.csv")
    if not (v1p.exists() and v2p.exists()):
        return                                     # manifests not built in this env; skip
    v1 = {Path(r["path"]).name: r["split"] for r in csv.DictReader(open(v1p))}
    seg = list(csv.DictReader(open(v2p)))
    seg_test = {r["basename"] for r in seg if r["split"] == "test"}
    seg_train = {r["basename"] for r in seg if r["split"] == "train"}
    v1_train = {b for b, s in v1.items() if s == "train"}
    v1_test = {b for b, s in v1.items() if s == "test"}
    assert not (seg_test & v1_train), "leakage: a V2 test image is in V1 train"
    assert not (seg_train & v1_test), "leakage: a V2 train image is in V1 test"
