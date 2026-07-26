"""Behavioural-index extraction — the vision modality the fusion model consumes.

Converts a short window of per-frame bird detections (+ track ids, + optional behaviour
class) into the five indices of ``poultryai.data.schema.VISION_CHANNELS``:

    activity_index            mean bird motion (tracked centroid speed), 0..1
    distribution_uniformity   how evenly birds occupy the floor (grid-occupancy entropy), 0..1
    piling_score              crowding / clumping (excess local density), 0..1
    feed_visit_rate           feeding intensity (eat-drink behaviour, or feeder-ROI presence), 0..1
    distress_index            composite early-distress proxy (immobility + sick + piling), 0..1

All indices are numpy-only, scale-free (normalised by image size / bird size), and bounded to
[0, 1] so they are directly usable as the fusion model's ``vision`` channel without extra
normalisation. Behavioural rationale (activity collapse, piling, reduced feeding, uneven
distribution are classic early welfare/health cues) is documented per function.

A ``Frame`` is one time step: boxes (N,4)=[x1,y1,x2,y2], optional track_ids (N,), optional
behaviour class ids (N,) using ``POSE_CLASSES`` indexing (0=eat-drink,1=moving,2=rest).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from poultryai.data.schema import VISION_CHANNELS
from . import POSE_CLASSES

_EAT_DRINK = POSE_CLASSES.index("eat-drink")   # behaviour-class id used by feed_visit_rate


@dataclass(slots=True)
class Frame:
    boxes: np.ndarray                                    # (N,4)
    track_ids: np.ndarray | None = None                  # (N,)
    behaviours: np.ndarray | None = None                 # (N,) pose-class ids, optional
    sick: np.ndarray | None = None                       # (N,) 0/1 healthy-sick, optional


def _centroids(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)


def _mean_bird_size(boxes: np.ndarray) -> float:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return 1.0
    w = (boxes[:, 2] - boxes[:, 0]).clip(0)
    h = (boxes[:, 3] - boxes[:, 1]).clip(0)
    return float(np.mean(np.sqrt(np.maximum(w * h, 1e-9))) or 1.0)


def activity_index(frames: list[Frame], image_wh: tuple[int, int]) -> float:
    """Mean tracked-centroid speed over the window, normalised by the image diagonal.

    Low activity is an early, disease-agnostic decline cue. Requires track_ids to link a
    bird across frames; returns 0 if no track can be followed for ≥2 frames.
    """
    diag = float(np.hypot(*image_wh)) or 1.0
    pos: dict[int, list[np.ndarray]] = {}
    for fr in frames:
        if fr.track_ids is None:
            continue
        c = _centroids(fr.boxes)
        for tid, ci in zip(np.asarray(fr.track_ids).reshape(-1), c):
            pos.setdefault(int(tid), []).append(ci)
    speeds = []
    for track in pos.values():
        if len(track) >= 2:
            d = np.diff(np.stack(track), axis=0)
            speeds.append(np.linalg.norm(d, axis=1).mean())
    if not speeds:
        return 0.0
    # normalise: a bird crossing ~10% of the frame diagonal per frame ≈ full activity
    return float(np.clip(np.mean(speeds) / (0.10 * diag), 0.0, 1.0))


def distribution_uniformity(frames: list[Frame], image_wh: tuple[int, int],
                            grid: int = 6) -> float:
    """Normalised occupancy entropy over a ``grid``×``grid`` floor tiling (1 = uniform).

    Birds crowding into part of the house (uneven distribution) is an early environmental/
    health signal (draughts, wet litter, illness). Averaged over frames.
    """
    W, H = image_wh
    vals = []
    for fr in frames:
        c = _centroids(fr.boxes)
        if len(c) == 0:
            continue
        gx = np.clip((c[:, 0] / max(W, 1) * grid).astype(int), 0, grid - 1)
        gy = np.clip((c[:, 1] / max(H, 1) * grid).astype(int), 0, grid - 1)
        occ = np.bincount(gy * grid + gx, minlength=grid * grid).astype(np.float64)
        p = occ / occ.sum()
        nz = p[p > 0]
        ent = -(nz * np.log(nz)).sum()
        vals.append(ent / np.log(grid * grid))     # normalise by max entropy
    return float(np.clip(np.mean(vals), 0.0, 1.0)) if vals else 0.0


def piling_score(frames: list[Frame], radius_factor: float = 1.5,
                 crowd_k: int = 3) -> float:
    """Fraction of birds with ≥``crowd_k`` neighbours within ``radius_factor``×bird-size.

    Piling (clumping) signals cold stress, panic, or sickness and precedes smothering. Uses
    a bird-size-relative radius so it is scale-free. Averaged over frames.
    """
    vals = []
    for fr in frames:
        c = _centroids(fr.boxes)
        n = len(c)
        if n < crowd_k + 1:
            vals.append(0.0); continue
        r = radius_factor * _mean_bird_size(fr.boxes)
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
        neigh = (d < r).sum(axis=1) - 1                 # exclude self
        vals.append(float((neigh >= crowd_k).mean()))
    return float(np.clip(np.mean(vals), 0.0, 1.0)) if vals else 0.0


def feed_visit_rate(frames: list[Frame], feeder_roi: tuple[float, float, float, float] | None = None
                    ) -> float:
    """Feeding intensity in [0,1].

    If behaviour labels exist (pose model), it is the mean fraction of birds classified
    ``eat-drink``. Else, if a ``feeder_roi`` (x1,y1,x2,y2) is given, it is the mean fraction of
    birds whose centroid falls inside it. Reduced feeding is an early illness cue (pairs with
    the water/feed sensor signals in the roadmap).
    """
    vals = []
    for fr in frames:
        n = len(np.asarray(fr.boxes).reshape(-1, 4))
        if n == 0:
            continue
        if fr.behaviours is not None:
            vals.append(float((np.asarray(fr.behaviours) == _EAT_DRINK).mean()))
        elif feeder_roi is not None:
            c = _centroids(fr.boxes)
            x1, y1, x2, y2 = feeder_roi
            inside = ((c[:, 0] >= x1) & (c[:, 0] <= x2) & (c[:, 1] >= y1) & (c[:, 1] <= y2))
            vals.append(float(inside.mean()))
    return float(np.clip(np.mean(vals), 0.0, 1.0)) if vals else 0.0


def distress_index(frames: list[Frame], image_wh: tuple[int, int],
                   w_immobility: float = 0.5, w_sick: float = 0.3, w_piling: float = 0.2
                   ) -> float:
    """Composite early-distress proxy = weighted(immobility, sick-fraction, piling), 0..1.

    A deliberately simple, interpretable combination of independent cues; the fusion model
    can learn to re-weight, but a standalone distress signal is a useful low-cost alarm.
    """
    immobility = 1.0 - activity_index(frames, image_wh)
    sick_vals = [float(np.asarray(fr.sick).mean()) for fr in frames if fr.sick is not None
                 and len(np.asarray(fr.sick).reshape(-1))]
    sick = float(np.mean(sick_vals)) if sick_vals else 0.0
    pile = piling_score(frames)
    total_w = w_immobility + w_sick + w_piling
    return float(np.clip((w_immobility * immobility + w_sick * sick + w_piling * pile) / total_w,
                         0.0, 1.0))


def window_indices(frames: list[Frame], image_wh: tuple[int, int],
                   feeder_roi: tuple[float, float, float, float] | None = None) -> np.ndarray:
    """Return the five indices in canonical ``VISION_CHANNELS`` order as a (5,) float array.

    This vector is exactly one ``vision`` timestep for ``poultryai.data.schema.Window``.
    """
    values = {
        "activity_index": activity_index(frames, image_wh),
        "distribution_uniformity": distribution_uniformity(frames, image_wh),
        "piling_score": piling_score(frames),
        "feed_visit_rate": feed_visit_rate(frames, feeder_roi),
        "distress_index": distress_index(frames, image_wh),
    }
    assert set(values) == set(VISION_CHANNELS), "index set must match schema.VISION_CHANNELS"
    return np.array([values[c] for c in VISION_CHANNELS], dtype=np.float32)
