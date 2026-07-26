"""Dependency-free multi-object tracker (SORT-lite, numpy only).

A greedy IoU-association tracker sufficient to turn per-frame detections into stable track
ids and per-track velocities — everything the behavioural-index extractor (``indices.py``)
needs. Deliberately dependency-free so it runs on the RPi4 edge and is unit-testable without
torch. Swap in ByteTrack/BoT-SORT later behind the same ``update`` interface if the tracking
benchmark (``roboflow_tracking``) warrants it.

Detections per frame are an (N, 4) array of ``[x1, y1, x2, y2]`` (pixels), optionally with a
parallel (N,) class-id array. ``update`` returns a list of :class:`Track` visible this frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a:(M,4) and b:(N,4) → (M,N)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


@dataclass(slots=True)
class Track:
    track_id: int
    box: np.ndarray                 # last [x1,y1,x2,y2]
    cls: int = -1
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))  # centroid px/frame
    age: int = 0                    # frames seen
    missed: int = 0                 # consecutive frames unmatched

    @property
    def centroid(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])


class SortLite:
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5, min_hits: int = 1):
        self.iou_threshold = iou_threshold
        self.max_age = max_age            # kill a track after this many missed frames
        self.min_hits = min_hits          # frames before a track is "confirmed"/returned
        self._next_id = 0
        self.tracks: list[Track] = []

    def update(self, boxes: np.ndarray, classes: np.ndarray | None = None) -> list[Track]:
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        classes = (np.asarray(classes).reshape(-1) if classes is not None
                   else np.full(len(boxes), -1))

        # greedy IoU association (highest IoU first)
        matched_det: set[int] = set()
        matched_tracks: set[int] = set()       # indices into self.tracks matched this frame
        if self.tracks and len(boxes):
            track_boxes = np.stack([t.box for t in self.tracks])
            iou = iou_matrix(track_boxes, boxes)
            pairs = sorted(((iou[i, j], i, j) for i in range(iou.shape[0])
                            for j in range(iou.shape[1]) if iou[i, j] >= self.iou_threshold),
                           reverse=True)
            for _, ti, dj in pairs:
                if ti in matched_tracks or dj in matched_det:
                    continue
                matched_tracks.add(ti); matched_det.add(dj)
                t = self.tracks[ti]
                new_c = np.array([(boxes[dj][0] + boxes[dj][2]) / 2,
                                  (boxes[dj][1] + boxes[dj][3]) / 2])
                t.velocity = new_c - t.centroid
                t.box = boxes[dj]; t.cls = int(classes[dj])
                t.age += 1; t.missed = 0

        # age tracks not matched this frame; drop the stale ones
        for ti, t in enumerate(self.tracks):
            if ti not in matched_tracks:
                t.missed += 1
        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]

        # spawn new tracks for unmatched detections
        for j in range(len(boxes)):
            if j not in matched_det:
                self.tracks.append(Track(track_id=self._next_id, box=boxes[j],
                                         cls=int(classes[j]), age=1, missed=0))
                self._next_id += 1

        return [t for t in self.tracks if t.missed == 0 and t.age >= self.min_hits]
