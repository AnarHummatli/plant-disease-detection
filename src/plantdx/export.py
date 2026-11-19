"""Inference model: the trained network without training-only parts (D-012, D-014).

The training checkpoint (best.keras) carries the optimizer state, the augmentation layers and,
under mixed precision, float16 compute layers. For evaluation timing, model size and the demo we
want the plain network:

* rebuilt in **float32** (CPUs have no fast float16 path);
* **without augmentation** (it is a no-op at inference anyway);
* **without optimizer state** (an uncompiled model is saved without it).

Weights are copied layer by layer by name, so the same code works for any model in
plantdx.models. The result, inference.keras, is what `size_mb` in the results table measures.

    python -m plantdx.export --run-id cnn_scratch      # for runs trained before D-014
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

INFERENCE_FILE = "inference.keras"


def copy_weights_by_name(source, target) -> int:
    """Copy weights from `source` layers into `target` layers with the same name.

    Every target layer that has weights must find a source layer with the same name and the same
    weight shapes; returns the number of layers copied.
    """
    by_name = {layer.name: layer for layer in source.layers}
    copied = 0
    for layer in target.layers:
        if not layer.weights:
            continue
        if layer.name not in by_name:
            raise ValueError(f"layer {layer.name!r} not found in the trained model")
        weights = by_name[layer.name].get_weights()
        if [w.shape for w in weights] != [tuple(v.shape) for v in layer.weights]:
            raise ValueError(f"weight shapes differ for layer {layer.name!r}")
        layer.set_weights(weights)
        copied += 1
    return copied


def build_inference_model(cfg: Mapping[str, Any], trained):
    """A float32, augmentation-free copy of `trained`, built from the same config."""
    import keras

    from plantdx.models import build_model

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
