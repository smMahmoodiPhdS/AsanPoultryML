"""Torch-free tests for V3: clip parsing/split, tracker, and behavioural indices."""
import numpy as np

from poultryai.data.schema import VISION_CHANNELS
from poultryai.tasks.vision.roboflow import (YoloImage, clip_id, clip_aware_split,
                                             assert_clip_split_clean)
from poultryai.tasks.vision.tracker import SortLite, iou_matrix
from poultryai.tasks.vision.indices import (Frame, window_indices, activity_index,
                                            piling_score, distribution_uniformity,
                                            feed_visit_rate)


# ---- roboflow clip parsing + split ----
def test_clip_id_strips_hash_and_frame_index():
    assert clip_id("A-chicken-crossing-the-road_mp4-12_jpg.rf.deadbeef01.jpg") \
        == "A-chicken-crossing-the-road"
    # frames of one clip collapse to the same id
    assert clip_id("vidA-0_jpg.rf.aaa.jpg") == clip_id("vidA-7_jpg.rf.bbb.jpg") == "vidA"
    # a purely-numeric frame name keeps its number so augmentations of it still group
    assert clip_id("01_jpg.rf.aaa.jpg") == clip_id("01_jpg.rf.bbb.jpg") == "01"


def test_clip_aware_split_no_leakage():
    imgs = []
    for clip in range(40):
        for frame in range(5):
            imgs.append(YoloImage(dataset="d", path=f"{clip}-{frame}.jpg", label_path="",
                                  orig_split="train", clip=f"clip{clip}", n_objects=1,
                                  class_hist=(1,)))
    counts = clip_aware_split(imgs, seed=1)
    assert_clip_split_clean(imgs)                 # raises on any clip spanning splits
    assert set(counts) == {"train", "val", "test"}
    # every frame of a clip shares one split
    by_clip = {}
    for im in imgs:
        by_clip.setdefault(im.clip, set()).add(im.split)
    assert all(len(s) == 1 for s in by_clip.values())


# ---- tracker ----
def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]]); b = np.array([[0, 0, 10, 10], [100, 100, 110, 110]])
    m = iou_matrix(a, b)
    assert abs(m[0, 0] - 1.0) < 1e-9 and m[0, 1] == 0.0


def test_tracker_keeps_stable_id_and_ages_out():
    tr = SortLite(iou_threshold=0.2, max_age=2)
    ids = [tr.update(np.array([[10 + k * 4, 10, 30 + k * 4, 30]]))[0].track_id for k in range(4)]
    assert len(set(ids)) == 1                     # one moving bird → one id
    # after disappearing, the track ages out and a new detection gets a fresh id
    for _ in range(3):
        tr.update(np.zeros((0, 4)))
    out = tr.update(np.array([[10, 10, 30, 30]]))
    assert out[0].track_id != ids[0]


# ---- indices ----
def test_indices_vector_shape_order_and_bounds():
    boxes = np.array([[x * 50, y * 50, x * 50 + 20, y * 50 + 20]
                      for x in range(3) for y in range(3)], float)
    frames = [Frame(boxes=boxes + k * 2, track_ids=np.arange(9)) for k in range(4)]
    v = window_indices(frames, (200, 200))
    assert v.shape == (5,) and v.dtype == np.float32
    assert ((v >= 0) & (v <= 1)).all()


def test_piling_and_uniformity_respond_to_clumping():
    clumped = np.array([[10, 10, 30, 30], [12, 12, 32, 32], [14, 11, 34, 31],
                        [11, 13, 31, 33], [13, 14, 33, 34]], float)
    fr = [Frame(boxes=clumped, track_ids=np.arange(5))]
    assert piling_score(fr) > 0.9
    assert distribution_uniformity(fr, (200, 200)) < 0.2


def test_activity_needs_tracks():
    # single frame → no motion measurable → activity 0
    fr = [Frame(boxes=np.array([[0, 0, 10, 10]]), track_ids=np.array([0]))]
    assert activity_index(fr, (100, 100)) == 0.0


def test_feed_visit_rate_from_behaviour():
    beh = np.array([0, 0, 1, 2])                  # 2 of 4 eat-drink (class 0)
    fr = [Frame(boxes=np.zeros((4, 4)), behaviours=beh)]
    assert abs(feed_visit_rate(fr) - 0.5) < 1e-9


def test_index_names_match_schema():
    fr = [Frame(boxes=np.array([[0, 0, 10, 10]]), track_ids=np.array([0]))]
    v = window_indices(fr, (100, 100))
    assert len(v) == len(VISION_CHANNELS) == 5
