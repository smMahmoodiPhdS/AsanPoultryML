"""Unit tests for the fecal leakage-safe split logic (numpy/PIL only — no torch needed).

These guard the single most important scientific property of V1: no image (exact or
grouped duplicate) may appear in more than one split. Mirrors the discipline in
``test_leadtime.py`` (numpy-only, fast, CI-friendly).
"""
import numpy as np

from poultryai.tasks.fecal import CLASS_TO_IDX
from poultryai.tasks.fecal.splits import (
    ImageRecord, assert_no_leakage, assign_groups, class_from_filename,
    stratified_group_split, _hamming,
)


def _rec(path, label, md5, ahash=0, is_pcr=False):
    return ImageRecord(path=path, label=label, is_pcr=is_pcr, md5=md5, ahash=ahash)


def test_class_from_filename():
    assert class_from_filename("cocci.12.jpg") == ("coccidiosis", False)
    assert class_from_filename("pcrncd.3.jpg") == ("newcastle", True)
    assert class_from_filename("salmo.9.jpg") == ("salmonella", False)


def test_exact_duplicates_share_a_group():
    recs = [_rec("a.jpg", "healthy", "MD5A"),
            _rec("b.jpg", "healthy", "MD5A"),   # identical bytes
            _rec("c.jpg", "healthy", "MD5B")]
    n = assign_groups(recs, max_hamming=0)
    assert n == 2
    assert recs[0].group_id == recs[1].group_id
    assert recs[2].group_id != recs[0].group_id


def test_near_dups_only_merge_within_class():
    # two images with identical aHash but different labels must NOT merge (collision guard)
    recs = [_rec("a.jpg", "healthy", "M1", ahash=0b1010),
            _rec("b.jpg", "coccidiosis", "M2", ahash=0b1010)]
    assign_groups(recs, max_hamming=2)
    assert recs[0].group_id != recs[1].group_id


def test_no_leakage_after_split():
    rng = np.random.default_rng(0)
    recs = []
    for i in range(400):
        cls = list(CLASS_TO_IDX)[i % 4]
        md5 = f"M{i//2}"                 # pairs of exact duplicates
        recs.append(_rec(f"{cls}.{i}.jpg", cls, md5))
    assign_groups(recs, max_hamming=0)
    counts = stratified_group_split(recs, val_frac=0.15, test_frac=0.15, seed=1)
    assert_no_leakage(recs)             # raises on any leak
    assert set(counts) == {"train", "val", "test"}
    # every duplicate pair landed in the same split
    by_md5 = {}
    for r in recs:
        by_md5.setdefault(r.md5, set()).add(r.split)
    assert all(len(s) == 1 for s in by_md5.values())


def test_stratification_preserves_classes():
    recs = []
    for i in range(800):
        cls = list(CLASS_TO_IDX)[i % 4]
        recs.append(_rec(f"{cls}.{i}.jpg", cls, f"M{i}"))
    assign_groups(recs, max_hamming=0)
    stratified_group_split(recs, seed=2)
    for split in ("train", "val", "test"):
        labels = [r.label for r in recs if r.split == split]
        # all four classes present in every split
        assert len(set(labels)) == 4


def test_hamming():
    assert _hamming(0b1111, 0b1011) == 1
    assert _hamming(0, 0) == 0
