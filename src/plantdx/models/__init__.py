"""Model builders, chosen by name in the experiment config (`model.name`).

Every model takes float32 images in 0-255 and returns softmax probabilities (D-008).
Stage 4 adds "mobilenetv2" here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

MODEL_NAMES = ("cnn",)


def build_model(cfg: Mapping[str, Any], augmentation=None):
    """Build the Keras model named by cfg["model"]["name"]."""
    name = cfg.get("model", {}).get("name")
    if name == "cnn":
        from plantdx.models.cnn import from_config

        return from_config(cfg, augmentation)
    raise ValueError(f"unknown model.name {name!r}; expected one of {MODEL_NAMES}")
