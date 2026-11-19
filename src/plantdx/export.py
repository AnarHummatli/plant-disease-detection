"""Inference model: the trained network without training-only parts (D-012, D-014).

The training checkpoint (best.keras) carries the optimizer state, the augmentation layers and,
under mixed precision, float16 compute layers. For evaluation timing, model size and the demo we
want the plain network:

* rebuilt in **float32** (CPUs have no fast float16 path);
* **without augmentation** (it is a no-op at inference anyway);
* **without optimizer state** (an uncompiled model is saved without it).

Weights are copied by their path ("<layer>/<variable>"), so the same code works for any model in
plantdx.models, including nested backbones with frozen layers. The result, inference.keras,
is what `size_mb` in the results table measures.

    python -m plantdx.export --run-id cnn_scratch      # for runs trained before D-014
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

INFERENCE_FILE = "inference.keras"


def copy_weights_by_name(source, target) -> int:
    """Copy every weight of `target` from the `source` weight with the same path.

    Paths are "<layer name>/<variable name>" (e.g. "block_13_expand/kernel"), unique within a
    model and independent of which layers are frozen, so the copy cannot silently shuffle
    weights between layers. Every target weight must exist in `source` with the same shape;
    source weights without a counterpart (e.g. augmentation state) are ignored. Returns the
    number of weights copied.
    """
    by_path = {w.path: w for w in source.weights}
    copied = 0
    for w in target.weights:
        if w.path not in by_path:
            raise ValueError(f"weight {w.path!r} not found in the trained model")
        src = by_path[w.path]
        if tuple(src.shape) != tuple(w.shape):
            raise ValueError(f"shape differs for {w.path!r}: {src.shape} vs {w.shape}")
        w.assign(src.numpy().astype(w.dtype))
        copied += 1
    return copied


def build_inference_model(cfg: Mapping[str, Any], trained):
    """A float32, augmentation-free copy of `trained`, built from the same config."""
    import keras

    from plantdx.models import build_model

    cfg = copy.deepcopy(dict(cfg))
    cfg.setdefault("model", {})["weights"] = None  # no ImageNet download: weights are copied
    policy = keras.mixed_precision.global_policy()
    keras.mixed_precision.set_global_policy("float32")
    try:
        model = build_model(cfg, augmentation=None)
    finally:
        keras.mixed_precision.set_global_policy(policy)
    copy_weights_by_name(trained, model)
    return model


def export_inference(cfg: Mapping[str, Any], trained, out_dir: str | Path) -> Path:
    """Write <out_dir>/inference.keras and return its path."""
    path = Path(out_dir) / INFERENCE_FILE
    build_inference_model(cfg, trained).save(path)
    return path


def export_run(run_id: str) -> Path:
    """Create inference.keras for an existing run folder from its best.keras and config.yaml."""
    import keras

    from plantdx.paths import run_dir

    folder = run_dir(run_id, create=False)
    cfg = yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8"))
    trained = keras.models.load_model(folder / "best.keras")
    path = export_inference(cfg, trained, folder)
    print(f"{path} ({path.stat().st_size / 1e6:.2f} MB)")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    export_run(parser.parse_args(argv).run_id)


if __name__ == "__main__":
    main()
