"""Torch-free tests for V4: YOLO-seg mask rasterisation, metrics, and split reuse."""
import csv
from pathlib import Path

import numpy as np

from poultryai.tasks.health_seg import N_SEG_CLASSES, SEG_CLASS_TO_IDX
from poultryai.tasks.health_seg.masks import parse_yoloseg_label, rasterise_yoloseg
from poultryai.tasks.health_seg.eval import (dominant_health, image_health_report,
                                             seg_metrics, update_confusion)


def test_rasterise_sick_polygon():
    # a normalised square polygon labelled sick (yolo class 1) → seg index for 'sick'
    polys = [(1, np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]))]
    mask = rasterise_yoloseg(polys, 100, 100)
    assert mask.max() == SEG_CLASS_TO_IDX["sick"]
    assert (mask == SEG_CLASS_TO_IDX["sick"]).sum() > 0
    assert mask.min() == 0                       # background remains


def test_rasterise_healthy_maps_to_class1():
    polys = [(0, np.array([[0.1, 0.1], [0.5, 0.1], [0.3, 0.5]]))]
    mask = rasterise_yoloseg(polys, 50, 50)
    assert mask.max() == SEG_CLASS_TO_IDX["healthy"] == 1


def test_empty_label_is_background():
    assert rasterise_yoloseg([], 20, 20).max() == 0


def test_dominant_health():
    m = np.zeros((10, 10), dtype=np.uint8)
    m[:4, :4] = SEG_CLASS_TO_IDX["sick"]
    assert dominant_health(m) == SEG_CLASS_TO_IDX["sick"]
    assert dominant_health(np.zeros((5, 5), dtype=np.uint8)) == 0


def test_seg_metrics_and_image_report_perfect():
    conf = np.zeros((N_SEG_CLASSES, N_SEG_CLASSES), dtype=np.int64)
    pred = np.array([[0, 1], [2, 0]])
    update_confusion(conf, pred, pred)
    m = seg_metrics(conf)
    assert abs(m["pixel_accuracy"] - 1.0) < 1e-9
    rep = image_health_report(np.array([1, 2, 0]), np.array([1, 2, 0]))
    assert abs(rep["accuracy"] - 1.0) < 1e-9


def test_split_reuse_leakage_free():
    """If the V3 manifest exists, V4 uses it and no clip may span splits."""
    mp = Path("poultryai/tasks/vision/manifests/roboflow_healthy_sick.csv")
    if not mp.exists():
        return
    rows = list(csv.DictReader(open(mp)))
    clip_splits = {}
    for r in rows:
        clip_splits.setdefault(r["clip"], set()).add(r["split"])
    assert all(len(s) == 1 for s in clip_splits.values()), "a clip spans multiple splits"


def test_parse_real_label_if_present():
    mp = Path("poultryai/tasks/vision/manifests/roboflow_healthy_sick.csv")
    if not mp.exists():
        return
    for r in csv.DictReader(open(mp)):
        polys = parse_yoloseg_label(r["label_path"])
        for cid, poly in polys:
            assert cid in (0, 1) and poly.shape[1] == 2 and len(poly) >= 3
        break
