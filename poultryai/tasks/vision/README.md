# V3 — Detection → tracking → behavioural indices

The upstream that makes the **vision modality** real. Its deliverable is not a SOTA detector
(the Roboflow sets are tiny) but the *pipeline* that turns a saloon camera stream into the five
behavioural indices the multimodal fusion model consumes
(`poultryai.data.schema.VISION_CHANNELS`). See
[`../../../docs/ai-model-plan.md`](../../../docs/ai-model-plan.md) (V3).

```
camera frames → YOLO detector → SORT-lite tracker → behavioural indices
                (transfer-learned)  (numpy, edge)     activity · distribution ·
                                                       piling · feeding · distress
```

## Datasets (all YOLO-format, `PhdThesis/Data/downloads/Roboflow/`)
| dataset | role | classes |
|---|---|---|
| `roboflow_broiler_detection` | train the bird detector | Chicken-Birds |
| `roboflow_tracking` | tracker eval (video clips) | rooster |
| `roboflow_chicken_pose` | behaviour → activity & feeding | eat-drink, moving, rest |
| `roboflow_healthy_sick` | a distress/health cue | Healthy, Sick |

## The leakage finding (why we re-split)
Roboflow ships random-over-frame splits, so **video frames from one clip leak across
train/val/test** — near-duplicates that inflate detection/tracking scores. `roboflow.py`
recovers each frame's **source clip id** and regroups whole clips into a leakage-free split.
Audit of the shipped splits (verified):

```
broiler_detection : 15/122 clips leaked        chicken_pose : 39/1371 clips leaked
tracking          : 3/15  clips leaked         healthy_sick : 11/75   clips leaked
```

Build the leakage-free manifests (run once, committed under `manifests/`):
```bash
python scripts/build_roboflow_manifest.py \
  --root PhdThesis/Data/downloads/Roboflow/Data/downloads \
  --out-dir poultryai/tasks/vision/manifests
```

## Layout
```
poultryai/tasks/vision/
  __init__.py   dataset ids, pose/health class names
  roboflow.py   YOLO scanner + clip-id parser + clip-aware split & audit  (torch-free)
  tracker.py    SORT-lite IoU tracker (numpy; edge-ready; swap ByteTrack)  (torch-free)
  indices.py    detections+tracks → 5 VISION_CHANNELS                      (torch-free)
  detect.py     Ultralytics YOLO train/infer wrappers                      (needs ultralytics)
  manifests/    committed clip-aware splits per dataset
scripts/
  build_roboflow_manifest.py    audit + clip-aware re-split (run once)
  train_detector.py             fine-tune YOLO on the leakage-free split
  extract_vision_indices.py     detector→tracker→indices → vision CSV
configs/vision.yaml
tests/test_vision.py            clip-split, tracker, indices tests (numpy-only)
```

## The five behavioural indices (`indices.py`)
All numpy, scale-free, bounded [0, 1], returned in canonical `VISION_CHANNELS` order:

| index | what it measures | early-warning rationale |
|---|---|---|
| `activity_index` | mean tracked-centroid speed / image diagonal | activity collapse precedes visible decline |
| `distribution_uniformity` | grid-occupancy entropy (1 = even) | uneven spread ⇒ draught / wet litter / illness |
| `piling_score` | fraction of birds with ≥k close neighbours | piling ⇒ cold stress / panic / smothering risk |
| `feed_visit_rate` | eat-drink fraction (pose) or feeder-ROI presence | reduced feeding is an early illness cue |
| `distress_index` | weighted(immobility, sick-fraction, piling) | composite low-cost distress alarm |

## Extract indices (runs now — no detector needed)
`extract_vision_indices.py` can use **ground-truth YOLO labels as detections**, so the
tracker+index pipeline is demonstrable directly on the Roboflow data:
```bash
# video clips → activity from real motion (single-class detector, no behaviour):
python scripts/extract_vision_indices.py --config configs/vision.yaml \
  --manifest poultryai/tasks/vision/manifests/roboflow_tracking.csv \
  --from-labels --out artifacts/vision/tracking_indices.csv

# pose data → real feeding/behaviour (--pose treats class ids as eat-drink/moving/rest):
python scripts/extract_vision_indices.py --config configs/vision.yaml \
  --manifest poultryai/tasks/vision/manifests/roboflow_chicken_pose.csv \
  --from-labels --pose --out artifacts/vision/pose_indices.csv
```
> `--pose` matters: a single-class **bird** detector's class id is *not* a behaviour, so
> without it `feed_visit_rate` correctly falls back to the feeder ROI (or 0), rather than
> mislabelling every bird as "eat-drink".

## Train the detector (needs ultralytics)
```bash
pip install ultralytics
python scripts/train_detector.py --config configs/vision.yaml   # trains on the clip-aware split
```
`train_detector.py` writes a leakage-free `data.yaml` from the manifest first, so reported mAP
is not inflated by frame leakage. YOLOv8-n (nano) is the edge default; export to ONNX for RPi4.

## Honest scope & limits
Small datasets ⇒ the detector is **transfer-learned** and framed as such; the tracking set has
only 15 source clips, so its split is coarse and tracking numbers are indicative. The pose set
is mostly single images, so it yields distribution/piling/feeding but not `activity_index`
(motion needs sequences) — the tracking videos supply activity. The scientific contribution is
the **stable, edge-deployable index pipeline** feeding the fusion model, not standalone mAP.
