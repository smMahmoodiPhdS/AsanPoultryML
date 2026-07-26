"""Minimal YAML config loader (avoids a hard Hydra dependency for simple runs)."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)
