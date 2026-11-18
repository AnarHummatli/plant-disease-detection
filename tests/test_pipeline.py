"""tf.data pipeline tests (plan 2.5 / 7.3). Skipped where TensorFlow is not installed."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

tf = pytest.importorskip("tensorflow")

from plantdx.config import load_config  # noqa: E402
from plantdx.data.pipeline import (  # noqa: E402
    AugmentConfig,
    build_augmentation,
    class_weights,
    datasets_from_config,
    load_split,
    make_dataset,
)

NO_AUG = AugmentConfig(flip="none", rotation_deg=0, zoom=0, brightness=0, contrast=0)
CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def write_images(root, n=20, size=(64, 48), mode="RGB", ext="jpg"):
    """n images of distinct classes 0..n-1; returns the split frame."""
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n):
        label = f"Crop{i:02d}___d"
        rel = f"{label}/img{i}.{ext}"
        path = root / "raw" / "color" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        Image.fromarray(arr).convert(mode).save(path)
        rows.append({"path": rel, "label": label, "label_index": i})
    return pd.DataFrame(rows)


def first_batch(ds):
    x, y = next(iter(ds))
    return x.numpy(), y.numpy()


def test_eval_batches_shapes_range_labels_and_order(tmp_path):
    frame = write_images(tmp_path)
    ds = make_dataset(frame, tmp_path, image_size=32, batch_size=8, num_classes=20)
    batches = list(ds)
    assert [b[0].shape[0] for b in batches] == [8, 8, 4]
    x, y = batches[0]
    assert x.dtype == tf.float32 and x.shape[1:] == (32, 32, 3)
    assert float(tf.reduce_min(x)) >= 0 and float(tf.reduce_max(x)) <= 255
    assert np.allclose(y.numpy().sum(axis=1), 1)
    labels = np.concatenate([b[1].numpy().argmax(1) for b in batches])
    assert labels.tolist() == frame["label_index"].tolist()  # eval keeps the frame order


def test_eval_is_deterministic(tmp_path):
    frame = write_images(tmp_path, n=6)
    ds = make_dataset(frame, tmp_path, image_size=32, batch_size=6, num_classes=6)
    a, _ = first_batch(ds)
    b, _ = first_batch(ds)
    assert np.array_equal(a, b)


def test_training_reshuffles_every_epoch(tmp_path):
    frame = write_images(tmp_path, n=20)
    ds = make_dataset(
        frame, tmp_path, image_size=16, batch_size=20, training=True, augment=NO_AUG, num_classes=20
    )
    orders = [first_batch(ds)[1].argmax(1).tolist() for _ in range(3)]
    assert sorted(orders[0]) == list(range(20))
    assert len({tuple(o) for o in orders}) > 1


def test_augmentation_changes_pixels_only_in_training(tmp_path):
    frame = write_images(tmp_path, n=1)
    plain, _ = first_batch(
        make_dataset(frame, tmp_path, image_size=32, batch_size=1, num_classes=1)
    )
    no_aug, _ = first_batch(
        make_dataset(
            frame,
            tmp_path,
            image_size=32,
            batch_size=1,
            training=True,
            augment=NO_AUG,
            num_classes=1,
        )
    )
    augmented, _ = first_batch(
        make_dataset(frame, tmp_path, image_size=32, batch_size=1, training=True, num_classes=1)
    )
    assert np.array_equal(plain, no_aug)
    assert not np.allclose(plain, augmented)
    assert augmented.min() >= 0 and augmented.max() <= 255


@pytest.mark.parametrize(("mode", "ext"), [("L", "jpg"), ("RGBA", "png"), ("RGB", "png")])
def test_decodes_grayscale_and_png_to_rgb(tmp_path, mode, ext):
    frame = write_images(tmp_path, n=2, size=(50, 30), mode=mode, ext=ext)
    x, _ = first_batch(make_dataset(frame, tmp_path, image_size=24, batch_size=2, num_classes=2))
    assert x.shape == (2, 24, 24, 3)


def test_memory_and_file_cache_give_same_images(tmp_path):
    frame = write_images(tmp_path, n=5)
    kw = dict(image_size=16, batch_size=5, num_classes=5)
    ref, _ = first_batch(make_dataset(frame, tmp_path, **kw))
    mem, _ = first_batch(make_dataset(frame, tmp_path, cache="memory", **kw))
    cache_file = tmp_path / "cache" / "eval"
    cache_file.parent.mkdir()
    ds = make_dataset(frame, tmp_path, cache=str(cache_file), **kw)
    first_batch(ds)
    list(ds)  # complete one pass so the cache file is finalised
    filed, _ = first_batch(ds)
    assert np.array_equal(ref, mem) and np.array_equal(ref, filed)
    assert any(cache_file.parent.iterdir())


def test_training_with_cache_still_shuffles_and_augments(tmp_path):
    frame = write_images(tmp_path, n=12)
    ds = make_dataset(
        frame, tmp_path, image_size=16, batch_size=12, training=True, cache="memory", num_classes=12
    )
    (x1, y1), (x2, _) = first_batch(ds), first_batch(ds)
    assert sorted(y1.argmax(1)) == list(range(12))
    assert not np.allclose(x1, x2)  # new random augmentation (and order) each epoch


def test_drop_remainder_and_empty_frame(tmp_path):
    frame = write_images(tmp_path, n=10)
    ds = make_dataset(
        frame, tmp_path, image_size=8, batch_size=4, drop_remainder=True, num_classes=10
    )
    assert len(list(ds)) == 2
    with pytest.raises(ValueError):
        make_dataset(frame.iloc[0:0], tmp_path)


def test_augmentation_layer_keeps_shape_and_range():
    aug = build_augmentation(AugmentConfig(), seed=1)
    x = tf.random.uniform((4, 32, 32, 3), 0, 255, seed=1)
    y = aug(x, training=True)
    assert y.shape == x.shape
    assert np.array_equal(aug(x, training=False).numpy(), x.numpy())  # identity at inference
    assert len(build_augmentation(NO_AUG).layers) == 0


def test_augment_config_from_dict_and_base_config():
    cfg = load_config(CONFIGS / "base.yaml")
    aug = AugmentConfig.from_dict(cfg["augment"])
    assert aug.rotation_deg == 20 and aug.flip == "horizontal_and_vertical"
    assert AugmentConfig.from_dict(None) == AugmentConfig()
    with pytest.raises(ValueError, match="unknown"):
        AugmentConfig.from_dict({"rotate": 10})


def test_class_weights_balanced():
    frame = pd.DataFrame({"label_index": [0, 0, 0, 1]})
    w = class_weights(frame, num_classes=2)
    assert w[0] == pytest.approx(4 / (2 * 3)) and w[1] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="without images"):
        class_weights(frame, num_classes=3)


def test_datasets_from_config(tmp_path):
    frame = write_images(tmp_path, n=6)
    splits = tmp_path / "splits"
    splits.mkdir()
    for name in ("train", "val", "test"):
        frame.to_csv(splits / f"{name}.csv", index=False)
    cfg = {
        "seed": 3,
        "data": {"image_size": 16, "num_classes": 6, "cache_eval": "memory"},
        "train": {"batch_size": 4},
        "augment": {"rotation_deg": 10},
    }
    ds = datasets_from_config(cfg, tmp_path, splits_dir=splits)
    assert set(ds) == {"train", "val", "test"}
    x, y = first_batch(ds["val"])
    assert x.shape == (4, 16, 16, 3) and y.shape == (4, 6)
    assert len(load_split("test", splits)) == 6
    with pytest.raises(ValueError):
        load_split("holdout", splits)


def test_plot_augmentations_writes_figure(tmp_path):
    from plantdx.viz import plot_augmentations

    frame = write_images(tmp_path, n=3, size=(64, 64))
    out = plot_augmentations(frame, tmp_path, "color", tmp_path / "aug.jpg", image_size=32, n_aug=3)
    assert out.stat().st_size > 1000
    no_aug = plot_augmentations(
        frame, tmp_path, "color", tmp_path / "aug0.jpg", augment=NO_AUG, image_size=32, n_aug=2
    )
    assert no_aug.exists()


def sorted_frame(root, n_classes=20, per_class=10):
    """Frame sorted by class, like splits/*.csv, with tiny images."""
    rows = []
    rng = np.random.default_rng(1)
    for c in range(n_classes):
        for k in range(per_class):
            rel = f"C{c:02d}___d/{k}.png"
            path = root / "raw" / "color" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)).save(path)
            rows.append({"path": rel, "label": f"C{c:02d}___d", "label_index": c})
    return pd.DataFrame(rows)


def test_cached_training_batches_mix_classes(tmp_path):
    """Regression: a small shuffle buffer over a class-sorted cache gave 1-2 classes per batch."""
    frame = sorted_frame(tmp_path)
    ds = make_dataset(
        frame,
        tmp_path,
        image_size=8,
        batch_size=20,
        training=True,
        augment=NO_AUG,
        cache="memory",
        shuffle_buffer=16,
        num_classes=20,
    )
    for _ in range(2):
        _, y = first_batch(ds)
        assert len(set(y.argmax(1))) >= 8


def test_disk_cache_named_by_size_and_file_list(tmp_path, monkeypatch):
    from plantdx.data.pipeline import cache_file

    monkeypatch.setenv("PLANTDX_CACHE", str(tmp_path / "cache"))
    a = cache_file(["x", "y"], 224, "train")
    assert a.parent.parent == tmp_path / "cache" and a.parent.name.startswith("train_224px_")
    assert cache_file(["x", "y"], 224, "train") == a
    assert cache_file(["x", "y"], 192, "train") != a
    assert cache_file(["x", "z"], 224, "train") != a

    frame = write_images(tmp_path, n=6)
    ds = make_dataset(
        frame,
        tmp_path,
        image_size=16,
        batch_size=3,
        training=True,
        augment=NO_AUG,
        cache="disk",
        cache_tag="train",
        num_classes=6,
    )
    epoch1 = sorted(float(x.numpy().sum()) for b, _ in ds for x in b)
    epoch2 = sorted(float(x.numpy().sum()) for b, _ in ds for x in b)
    assert epoch1 == epoch2 and len(epoch1) == 6
    assert any((tmp_path / "cache").rglob("cache*"))


def test_augment_on_model_keeps_pipeline_plain(tmp_path):
    from plantdx.data.pipeline import augment_on, augmentation_from_config

    frame = write_images(tmp_path, n=4)
    splits = tmp_path / "splits"
    splits.mkdir()
    for name in ("train", "val", "test"):
        frame.to_csv(splits / f"{name}.csv", index=False)
    base = {"data": {"image_size": 16, "num_classes": 4}, "train": {"batch_size": 4}}

    ds = datasets_from_config(base, tmp_path, splits_dir=splits)  # default: augment_on model
    plain = {round(float(x.sum()), 3) for x in first_batch(ds["val"])[0]}
    train = {round(float(x.sum()), 3) for x in first_batch(ds["train"])[0]}
    assert train == plain  # no augmentation in tf.data
    assert augment_on(base) == "model"
    assert augmentation_from_config(base) is not None

    cfg = {**base, "train": {"batch_size": 4, "augment_on": "pipeline"}}
    assert augmentation_from_config(cfg) is None
    train_aug = {
        round(float(x.sum()), 3)
        for x in first_batch(datasets_from_config(cfg, tmp_path, splits_dir=splits)["train"])[0]
    }
    assert train_aug != plain

    off = {
        **base,
        "augment": {"flip": "none", "rotation_deg": 0, "zoom": 0, "brightness": 0, "contrast": 0},
    }
    assert augmentation_from_config(off) is None
    with pytest.raises(ValueError, match="augment_on"):
        augment_on({"train": {"augment_on": "gpu"}})
