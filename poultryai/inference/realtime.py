"""Streaming edge inference on the Raspberry Pi 4.

Subscribes to live sensor values over MQTT, maintains rolling per-saloon windows, runs
the model at a fixed cadence, and publishes per-disease early-warning scores back to
MQTT for Node-RED / Grafana / the mobile app to consume.

Topic conventions mirror Docs/Standards/mqtt-topic-structure.md. Predictions are
published under an ``ai/`` root so they are cleanly separable from raw sensor data:

    ai/{owner}/{saloon}/earlywarning/{disease}   ->  score in [0,1]  (retained)
    ai/{owner}/{saloon}/earlywarning/alarm       ->  JSON {disease, score, ts}

This module is intentionally dependency-light (paho-mqtt + onnxruntime) so it runs on
the edge without the full training stack.
"""
from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..data.schema import DISEASES, ENV_CHANNELS


@dataclass
class EdgeConfig:
    owner: str
    saloon: str
    broker_host: str = "localhost"
    broker_port: int = 1883
    window_minutes: int = 240        # must match training window
    infer_period_s: int = 60
    alarm_threshold: float = 0.5
    persistence: int = 3


class RollingWindow:
    """Fixed-length ring of the latest environmental samples (1/min)."""
    def __init__(self, minutes: int) -> None:
        self.buf: deque[np.ndarray] = deque(maxlen=minutes)

    def push(self, sample: np.ndarray) -> None:
        self.buf.append(sample)

    def ready(self) -> bool:
        return len(self.buf) == self.buf.maxlen

    def array(self) -> np.ndarray:
        return np.stack(self.buf, axis=0)


class EdgeInference:
    """Glue: MQTT in -> ONNX model -> MQTT out. Model + audio/vision providers injected
    so the class is testable without a live broker."""

    def __init__(self, cfg: EdgeConfig, onnx_session, audio_provider, vision_provider,
                 meta_provider) -> None:
        self.cfg = cfg
        self.sess = onnx_session
        self.audio_provider = audio_provider     # () -> np.ndarray waveform
        self.vision_provider = vision_provider   # () -> np.ndarray (T_vis, C_vis)
        self.meta_provider = meta_provider       # () -> np.ndarray (meta_dim,)
        self.window = RollingWindow(cfg.window_minutes)
        self._run = np.zeros(len(DISEASES), dtype=int)

    # --- MQTT callbacks -----------------------------------------------------
    def on_sensor(self, node: str, value: float) -> None:
        """Map an incoming curvalue to its env channel and append when a minute completes.
        A real deployment buffers each channel and snapshots once per minute; simplified
        here to show the contract."""
        # (channel mapping handled by the caller; omitted for brevity)

    # --- inference ----------------------------------------------------------
    def infer(self) -> dict[str, float] | None:
        if not self.window.ready():
            return None
        env = self.window.array()[None].astype(np.float32)         # (1,T,C)
        audio = self.audio_provider()[None].astype(np.float32)
        vision = self.vision_provider()[None].astype(np.float32)
        meta = self.meta_provider()[None].astype(np.float32)
        present = np.ones((1, 3), dtype=bool)
        logits = self.sess.run(["logits"], {
            "env": env, "audio": audio, "vision": vision,
            "meta": meta, "present": present})[0]
        probs = 1.0 / (1.0 + np.exp(-logits[0]))
        return {d.value: float(p) for d, p in zip(DISEASES, probs)}

    def step_and_publish(self, publish) -> None:
        scores = self.infer()
        if scores is None:
            return
        base = f"ai/{self.cfg.owner}/{self.cfg.saloon}/earlywarning"
        for i, (name, score) in enumerate(scores.items()):
            publish(f"{base}/{name}", f"{score:.4f}", retain=True)
            self._run[i] = self._run[i] + 1 if score >= self.cfg.alarm_threshold else 0
            if self._run[i] == self.cfg.persistence:
                publish(f"{base}/alarm", json.dumps(
                    {"disease": name, "score": round(score, 4), "ts": int(time.time())}))
