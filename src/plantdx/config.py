"""YAML experiment configs with single-level inheritance, plus global seeding.

An experiment config can start with `extends: base.yaml`; its keys are deep-merged
on top of the parent, so each experiment file only states what it changes.
"""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict: `override` merged into `base`, recursing into nested dicts."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving `extends:` relative to the file's folder."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_merge(load_config(path.parent / parent), cfg)
    return cfg


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and (if installed) TensorFlow/Keras for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras

        keras.utils.set_random_seed(seed)
    except ImportError:
        pass
