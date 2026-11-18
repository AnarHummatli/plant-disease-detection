"""Classical baseline: features and linear models (plan steps 3.1-3.2, test 3.4)."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from plantdx.baselines.classical import fit_grid, make_model, scores
from plantdx.baselines.features import (
    LBP_BINS,
    cache_name,
    feature_dim,
    feature_slices,
    image_features,
    load_rgb,
    split_features,
)


def rgb(seed=0, size=128):
    return np.random.default_rng(seed).integers(0, 256, (size, size, 3), dtype=np.uint8)


def test_feature_size_and_layout():
    v = image_features(rgb())
    assert v.dtype == np.float32
    assert v.shape == (feature_dim(128),) == (1870,)
    sl = feature_slices(128)
    assert sl["hsv"].stop - sl["hsv"].start == 96
    assert sl["hog"].stop - sl["hog"].start == 1764
    assert sl["lbp"].stop - sl["lbp"].start == LBP_BINS == 10
    assert sl["lbp"].stop == len(v)


def test_histograms_are_normalised():
    v = image_features(rgb(1))
    sl = feature_slices(128)
    hsv = v[sl["hsv"]].reshape(3, 32)
    assert np.allclose(hsv.sum(axis=1), 1.0, atol=1e-5)
    assert v[sl["lbp"]].sum() == pytest.approx(1.0, abs=1e-5)
    assert np.all(v >= 0)


def test_features_are_deterministic_and_discriminative():
    a, b = image_features(rgb(2)), image_features(rgb(2))
    assert np.array_equal(a, b)
    green = np.zeros((128, 128, 3), np.uint8)
    green[..., 1] = 180
    brown = np.zeros((128, 128, 3), np.uint8)
    brown[...] = (120, 70, 20)
    hsv = feature_slices(128)["hsv"]
    assert not np.allclose(image_features(green)[hsv], image_features(brown)[hsv])


def test_feature_dim_other_sizes():
    for size in (64, 96, 160):
        assert image_features(rgb(size=size)).shape == (feature_dim(size),)


def test_rejects_wrong_input():
    with pytest.raises(ValueError):
        image_features(rgb().astype(np.float32))
    with pytest.raises(ValueError):
        image_features(rgb()[..., 0])


def test_load_rgb_converts_mode_and_resizes(tmp_path):
    Image.fromarray(rgb(size=256)).convert("RGBA").save(tmp_path / "a.png")
    Image.fromarray(rgb(size=256)[..., 0]).save(tmp_path / "g.jpg")
    for name in ("a.png", "g.jpg"):
        img = load_rgb(tmp_path / name, size=128)
        assert img.shape == (128, 128, 3) and img.dtype == np.uint8


def write_split(root, n_per_class=3, classes=3, size=64):
    rows = []
    for c in range(classes):
        for j in range(n_per_class):
            rel = f"C{c}___d/img{j}.jpg"
            p = root / "raw" / "color" / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb(seed=10 * c + j, size=size)).save(p)
            rows.append({"path": rel, "label_index": c})
    return pd.DataFrame(rows)


def test_split_features_cache_roundtrip(tmp_path):
    frame = write_split(tmp_path)
    cache = tmp_path / "cache"
    X, y = split_features(frame, tmp_path, size=64, cache_dir=cache, tag="train", n_jobs=1)
    assert X.shape == (9, feature_dim(64)) and y.tolist() == frame["label_index"].tolist()
    files = list(cache.glob("*.npz"))
    assert [f.name for f in files] == [cache_name(frame["path"].tolist(), 64, "train")]
    # second call reads the cache, even with the images gone
    for f in (tmp_path / "raw").rglob("*.jpg"):
        f.unlink()
    X2, y2 = split_features(frame, tmp_path, size=64, cache_dir=cache, tag="train", n_jobs=1)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)


def test_cache_name_changes_with_file_list_and_size():
    a = cache_name(["x", "y"], 128)
    assert a != cache_name(["x", "z"], 128)
    assert a != cache_name(["x", "y"], 64)


def blobs(n=60, d=1870, k=3, seed=0):
    """Train and val samples drawn around the same k class centres."""
    centres = np.random.default_rng(123).normal(size=(k, d)) * 0.5
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(k), n // k)
    return (centres[y] + rng.normal(size=(len(y), d))).astype(np.float32), y


@pytest.mark.parametrize("name", ["logreg", "linsvm"])
def test_fit_grid_selects_by_val_macro_f1(name):
    X, y = blobs()
    Xv, yv = blobs(seed=1)
    res = fit_grid(name, X, y, Xv, yv, Cs=[10.0, 0.1], pca=8, verbose=False)
    assert [t["C"] for t in res.trials] == [0.1, 10.0]  # sorted
    best = max(res.trials, key=lambda t: t["val_macro_f1"])
    assert res.best_C == best["C"]
    assert res.best_val["macro_f1"] == best["val_macro_f1"] > 0.9
    assert scores(res.best_model, Xv).shape == (len(yv), 3)


def test_pca_only_touches_hog_block():
    model = make_model("logreg", 1.0, pca=8)
    X, y = blobs(d=1870)
    model.fit(X, y)
    reduce = model.steps[0][1]
    # 8 HOG components + 96 colour + 10 LBP values reach the classifier
    assert reduce.transform(X[:2]).shape == (2, 8 + 96 + 10)


def test_unknown_model():
    with pytest.raises(ValueError):
        make_model("forest", 1.0)
