"""YOLO detector wrappers (needs `ultralytics` at runtime).

Thin, dependency-isolated adapters around Ultralytics YOLO so the rest of the pipeline
(tracker, indices) stays framework-free. Two roles:

* :func:`train_yolo` — fine-tune a pretrained YOLO on a Roboflow ``data.yaml`` (the small sets
  in this project → transfer learning, not from scratch).
* :class:`YoloDetector` — run inference on frames and return boxes/classes/scores in the plain
  numpy form the tracker expects. Exports to ONNX for RPi4/VPS serving.

Kept out of the import path of ``indices``/``tracker`` so those remain testable without torch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def train_yolo(data_yaml: str | Path, model: str = "yolov8n.pt", epochs: int = 100,
               imgsz: int = 640, project: str = "artifacts/vision", name: str = "detector",
               **kw):
    """Fine-tune a YOLO detector. Returns the ultralytics results object.

    Use the **clip-aware** re-split (see ``roboflow.py`` / ``build_roboflow_manifest.py``) —
    write a data.yaml pointing at the leakage-free train/val/test image lists before training,
    otherwise reported mAP is inflated by video-frame leakage.
    """
    from ultralytics import YOLO
    net = YOLO(model)
    return net.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz,
                     project=project, name=name, **kw)


class YoloDetector:
    """Inference adapter: frame(s) → (boxes, classes, scores) numpy arrays."""

    def __init__(self, weights: str | Path, conf: float = 0.25, imgsz: int = 640):
        from ultralytics import YOLO
        self.net = YOLO(str(weights))
        self.conf = conf
        self.imgsz = imgsz

    def detect(self, image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return boxes (N,4)=[x1,y1,x2,y2], classes (N,), scores (N,) for one image."""
        r = self.net.predict(image, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return (np.zeros((0, 4)), np.zeros((0,), dtype=int), np.zeros((0,)))
        return (r.boxes.xyxy.cpu().numpy(),
                r.boxes.cls.cpu().numpy().astype(int),
                r.boxes.conf.cpu().numpy())

    def export_onnx(self, imgsz: int | None = None):
        return self.net.export(format="onnx", imgsz=imgsz or self.imgsz)
