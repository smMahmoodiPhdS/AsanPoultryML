"""V3 — Detection → tracking → behavioural indices (the vision encoder's upstream).

The scientific product of V3 is **not** a state-of-the-art detector (the Roboflow sets are
small — hundreds of images). It is the *pipeline* that turns a saloon camera stream into the
five behavioural indices the multimodal fusion model consumes
(``poultryai.data.schema.VISION_CHANNELS``):

    detector (YOLO, transfer-learned)  →  multi-object tracker  →  behavioural indices
                                                                    (activity, distribution,
                                                                     piling, feeding, distress)

Datasets (``PhdThesis/Data/downloads/Roboflow/``), all YOLO-format:
* ``roboflow_broiler_detection`` — bird bounding boxes (the detector's training data).
* ``roboflow_tracking``          — rooster boxes from video clips (tracker evaluation).
* ``roboflow_chicken_pose``      — eat-drink / moving / rest (behaviour → activity & feeding).
* ``roboflow_healthy_sick``      — Healthy / Sick instance masks (a distress/health cue).

Modules
-------
* ``roboflow`` — YOLO dataset scanner + **clip-aware** re-split & leakage audit (torch-free).
  Roboflow's shipped splits leak video frames across train/val/test; this regroups by source
  clip so detection/tracking metrics are not inflated (``docs/ai-model-plan.md`` §6.1).
* ``tracker``  — dependency-free SORT-lite IoU tracker (numpy). Swappable for ByteTrack.
* ``indices``  — detections + tracks → the five VISION_CHANNELS (numpy; unit-tested).
* ``detect``   — ultralytics-YOLO train/infer wrappers (needs ultralytics at runtime).
"""
from __future__ import annotations

# canonical dataset ids
DATASETS: tuple[str, ...] = (
    "roboflow_broiler_detection",
    "roboflow_tracking",
    "roboflow_chicken_pose",
    "roboflow_healthy_sick",
)

# behaviour classes in the pose dataset (index = YOLO class id)
POSE_CLASSES: tuple[str, ...] = ("eat-drink", "moving", "rest")
HEALTH_CLASSES: tuple[str, ...] = ("Healthy", "Sick")
