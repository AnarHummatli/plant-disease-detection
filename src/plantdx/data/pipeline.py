"""tf.data input pipeline (plan step 2.5).

    [train: fixed shuffle of file names] -> read file -> decode (RGB) -> resize to S x S
    -> [cache, uint8] -> [train: shuffle] -> batch -> float32 in [0, 255]
    -> [augment, only if augment_on == "pipeline"] -> one-hot labels -> prefetch

Design choices (D-008):

* Images leave the pipeline as float32 in the 0-255 range. Each model does its own
  preprocessing (e.g. MobileNetV2's [-1, 1] scaling) inside the model, so the demo and the
  exported models can take raw pixels.
* Labels are one-hot, for CategoricalCrossentropy (and label smoothing later).
* Augmentation (random flips, rotation, zoom, brightness, contrast; seeded Keras layers) runs
  by default **on the GPU as the first layers of the model** (`augment_on: model`): measured
  on Colab (2 CPU cores), doing it in tf.data limited training input to 67 images/s.
  `augmentation_from_config` builds those layers; Keras switches them off at inference.
* Decoded images are cached as uint8 (a 224x224x3 image is 147 KB; float32 would be 4x):
  val/test in memory, train (about 5.5 GB) in a file on local disk (`cache: disk`). Cache file
  names contain the image size and a hash of the file list, so a changed split never reuses a
  stale cache.
* Training order: the file list is shuffled once with the seed **before** decoding/caching
  (the CSVs are sorted by class), then a buffer shuffle runs every epoch on top.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plantdx.data.images import image_path
from plantdx.data.source import CLASSES
from plantdx.paths import cache_dir, repo_root

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class AugmentConfig:
    """Training-time augmentation. Factors follow the Keras preprocessing layers."""

    flip: str = (
        "horizontal_and_vertical"  # "horizontal", "vertical", "horizontal_and_vertical" or "none"
    )
    rotation_deg: float = 20.0  # maximum rotation in degrees, both directions
    zoom: float = 0.15  # zoom in/out by up to this fraction
    brightness: float = 0.15  # shift by up to this fraction of the 0-255 range
    contrast: float = 0.15  # contrast factor drawn from [1 - c, 1 + c]

    @classmethod
    def disabled(cls) -> AugmentConfig:
        return cls(flip="none", rotation_deg=0, zoom=0, brightness=0, contrast=0)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> AugmentConfig:
        values = dict(values or {})
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(f"unknown augmentation keys: {sorted(unknown)}")
        return cls(**values)


def load_split(split: str, splits_dir: str | Path | None = None) -> pd.DataFrame:
    """Read splits/<split>.csv (path, label, label_index, leaf_id, group_id, leaf_source)."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    splits_dir = Path(splits_dir) if splits_dir else repo_root() / "splits"
    return pd.read_csv(splits_dir / f"{split}.csv")


def class_weights(frame: pd.DataFrame, num_classes: int = len(CLASSES)) -> dict[int, float]:
    """Balanced class weights n / (k * n_c), for an optional class-weighted loss."""
    counts = np.bincount(frame["label_index"].to_numpy(), minlength=num_classes)
    if (counts == 0).any():
        raise ValueError(f"classes without images: {np.flatnonzero(counts == 0).tolist()}")
    weights = len(frame) / (num_classes * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def build_augmentation(cfg: AugmentConfig | None = None, seed: int = 42):
    """Keras Sequential applying `cfg` to float32 batches in [0, 255]; active when training=True."""
    import keras

    cfg = cfg or AugmentConfig()
    layers = []
    if cfg.flip != "none":
        layers.append(keras.layers.RandomFlip(cfg.flip, seed=seed))
    if cfg.rotation_deg:
        layers.append(
            keras.layers.RandomRotation(
                cfg.rotation_deg / 360.0, fill_mode="reflect", seed=seed + 1
            )
        )
    if cfg.zoom:
        layers.append(
            keras.layers.RandomZoom((-cfg.zoom, cfg.zoom), fill_mode="reflect", seed=seed + 2)
        )
    if cfg.brightness:
        layers.append(
            keras.layers.RandomBrightness(cfg.brightness, value_range=(0, 255), seed=seed + 3)
        )
    if cfg.contrast:
        layers.append(
            keras.layers.RandomContrast(cfg.contrast, value_range=(0, 255), seed=seed + 4)
        )
    return keras.Sequential(layers, name="augmentation")


def _decode(path, image_size: int):
    import tensorflow as tf

    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, (image_size, image_size), method="bilinear", antialias=True)
    img = tf.cast(tf.round(tf.clip_by_value(img, 0.0, 255.0)), tf.uint8)
    img.set_shape((image_size, image_size, 3))
    return img


def cache_file(paths: list[str], image_size: int, tag: str = "") -> Path:
    """Disk cache prefix under cache_dir(): <tag>_<size>px_<hash of the file list>."""
    digest = hashlib.sha1("\n".join(paths).encode()).hexdigest()[:12]
    name = f"{tag + '_' if tag else ''}{image_size}px_{digest}"
    return cache_dir() / name / "cache"


def make_dataset(
    frame: pd.DataFrame,
    root: str | Path,
    variant: str = "color",
    image_size: int = 224,
    batch_size: int = 64,
    training: bool = False,
    augment: AugmentConfig | None = None,
    seed: int = 42,
    cache: str | None = None,
    cache_tag: str = "",
    shuffle_buffer: int = 4096,
    num_classes: int = len(CLASSES),
    drop_remainder: bool = False,
):
    """Batched dataset of (images float32 [B, S, S, 3] in 0-255, one-hot labels [B, C]).

    `training=True` shuffles every epoch and applies `augment` in the pipeline (default
    AugmentConfig(); pass AugmentConfig.disabled() when augmentation runs in the model).
    `cache` is None, "memory", "disk" (a file under cache_dir(), named by `cache_tag`, size and
    file-list hash) or an explicit file prefix. Caches hold decoded uint8 images, so
    augmentation stays random every epoch.
    """
    import tensorflow as tf

    autotune = tf.data.AUTOTUNE
    paths = [str(image_path(root, variant, p)) for p in frame["path"]]
    labels = frame["label_index"].to_numpy(dtype=np.int32)
    if len(paths) == 0:
        raise ValueError("empty frame")

    if training and cache:
        # The CSVs are sorted by class and a cache replays one fixed order, so mix the file
        # list once here; the per-epoch buffer shuffle below then works on a mixed stream.
        order = np.random.default_rng(seed).permutation(len(paths))
        paths = [paths[i] for i in order]
        labels = labels[order]

    if cache == "disk":
        prefix = cache_file(paths, image_size, cache_tag)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        cache = str(prefix)
    elif cache not in (None, "memory"):
        Path(cache).parent.mkdir(parents=True, exist_ok=True)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training and not cache:
        # shuffling file names is cheap, so the whole split is one shuffle buffer
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, y: (_decode(p, image_size), y),
        num_parallel_calls=autotune,
        deterministic=not training,
    )
    if cache:
        ds = ds.cache("" if cache == "memory" else str(cache))
        if training:
            ds = ds.shuffle(
                min(shuffle_buffer, len(paths)), seed=seed, reshuffle_each_iteration=True
            )
    ds = ds.batch(batch_size, drop_remainder=drop_remainder)

    aug = build_augmentation(augment, seed=seed) if training else None
    if aug is not None and not aug.layers:  # every factor is 0: nothing to apply
        aug = None

    def finish(x, y):
        x = tf.cast(x, tf.float32)
        if aug is not None:
            x = tf.clip_by_value(aug(x, training=True), 0.0, 255.0)
        return x, tf.one_hot(y, num_classes)

    ds = ds.map(finish, num_parallel_calls=autotune, deterministic=not training)
    return ds.prefetch(autotune)


def augment_on(cfg: Mapping[str, Any]) -> str:
    where = cfg.get("train", {}).get("augment_on", "model")
    if where not in ("model", "pipeline"):
        raise ValueError(f"train.augment_on must be 'model' or 'pipeline', not {where!r}")
    return where


def augmentation_from_config(cfg: Mapping[str, Any]):
    """The augmentation layers to put at the start of the model, or None.

    None when `train.augment_on` is "pipeline" (augmentation already happens in tf.data) or
    when every factor is 0.
    """
    if augment_on(cfg) != "model":
        return None
    aug = build_augmentation(AugmentConfig.from_dict(cfg.get("augment")), seed=cfg.get("seed", 42))
    return aug if aug.layers else None


def datasets_from_config(
    cfg: Mapping[str, Any], root: str | Path, splits_dir: str | Path | None = None
) -> dict:
    """train / val / test datasets with the settings of an experiment config."""
    data, train = cfg.get("data", {}), cfg.get("train", {})
    common = {
        "root": root,
        "variant": data.get("variant", "color"),
        "image_size": data.get("image_size", 224),
        "batch_size": train.get("batch_size", 64),
        "seed": cfg.get("seed", 42),
        "num_classes": data.get("num_classes", len(CLASSES)),
    }
    augment = (
        AugmentConfig.from_dict(cfg.get("augment"))
        if augment_on(cfg) == "pipeline"
        else AugmentConfig.disabled()
    )
    return {
        "train": make_dataset(
            load_split("train", splits_dir),
            training=True,
            augment=augment,
            cache=data.get("cache_train"),
            cache_tag="train",
            **common,
        ),
        **{
            split: make_dataset(
                load_split(split, splits_dir),
                cache=data.get("cache_eval"),
                cache_tag=split,
                **common,
            )
            for split in ("val", "test")
        },
    }
