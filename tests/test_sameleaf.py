"""Tests for the same-leaf recovery (plantdx.data.sameleaf, D-010)."""

import numpy as np
import pandas as pd
import pytest

from plantdx.data.sameleaf import (
    annotate,
    calibration_curve,
    group_size_stats,
    knn_pairs,
    load_embeddings,
    pick_threshold,
    save_embeddings,
)
from plantdx.data.split import load_group_pairs, merge_duplicate_groups


def clustered(n_leaves=6, per_leaf=4, dim=16, noise=0.05, seed=0, label=0, prefix="c0"):
    """Embeddings where photos of one leaf are close; returns (table, emb)."""
    rng = np.random.default_rng(seed)
    rows, embs = [], []
    for leaf in range(n_leaves):
        centre = rng.normal(size=dim)
        for k in range(per_leaf):
            e = centre + noise * rng.normal(size=dim)
            embs.append(e / np.linalg.norm(e))
            rows.append(
                {
                    "path": f"{prefix}/L{leaf}_{k}.jpg",
                    "label_index": label,
                    "leaf_id": f"{prefix}:{leaf}",
                    "leaf_source": "map",
                    "split": "train" if k < 3 else "test",
                }
            )
    return pd.DataFrame(rows), np.asarray(embs, dtype=np.float32)


def test_knn_pairs_finds_same_leaf_neighbours_within_class():
    t0, e0 = clustered(label=0, prefix="c0", seed=1)
    t1, e1 = clustered(label=1, prefix="c1", seed=2)
    table = pd.concat([t0, t1], ignore_index=True)
    emb = np.vstack([e0, e1])
    pairs = knn_pairs(table, emb, k=3, chunk=5)
    assert (pairs["path_a"].str[:2] == pairs["path_b"].str[:2]).all()  # never across classes
    assert not (pairs["path_a"] == pairs["path_b"]).any()
    assert not pairs.duplicated(["path_a", "path_b"]).any()
    ann = annotate(pairs, table)
    # with tight clusters of 4, the 3 nearest neighbours are the other photos of the leaf
    assert ann["same_leaf"].all() and ann["mutual"].all()
    assert len(pairs) == 2 * 6 * 6  # 6 pairs per leaf, 6 leaves, 2 classes


def test_calibration_curve_precision_and_recall():
    table = pd.DataFrame(
        {
            "path": ["a1", "a2", "b1", "f1"],
            "leaf_id": ["A", "A", "B", "fallback_f1"],
            "leaf_source": ["map", "map", "map", "fallback"],
            "split": ["train", "test", "train", "test"],
        }
    )
    pairs = pd.DataFrame(
        {
            "path_a": ["a1", "a1", "a2", "a1"],
            "path_b": ["a2", "b1", "b1", "f1"],
            "label_index": 0,
            "sim": [0.9, 0.8, 0.6, 0.95],
            "mutual": [True, True, False, True],
        }
    )
    ann = annotate(pairs, table)
    assert ann["involves_fallback"].tolist() == [False, False, False, True]
    curve = calibration_curve(ann, table, thresholds=[0.5, 0.7, 0.85]).set_index("threshold")
    assert curve.loc[0.5, "pairs"] == 3 and curve.loc[0.5, "precision"] == pytest.approx(
        1 / 3, abs=1e-3
    )
    assert curve.loc[0.85, "precision"] == 1.0 and curve.loc[0.85, "recall"] == 1.0
    assert curve.loc[0.7, "recall"] == 1.0  # the only same-leaf pair (a1, a2) is found


def test_pick_threshold_requires_precision_to_stay_high():
    curve = pd.DataFrame(
        {"threshold": [0.5, 0.6, 0.7, 0.8, 0.9], "precision": [0.5, 0.96, 0.90, 0.97, 0.99]}
    )
    assert pick_threshold(curve, 0.95) == 0.8  # 0.6 is above the bar, but 0.7 dips below
    assert pick_threshold(curve, 0.995) is None


def test_group_size_stats_and_merge():
    table = pd.DataFrame(
        {
            "path": ["f1", "f2", "f3", "f4"],
            "leaf_id": ["fallback_1", "fallback_2", "fallback_3", "fallback_4"],
            "leaf_source": ["fallback"] * 4,
        }
    )
    pairs = pd.DataFrame({"path_a": ["f1", "f2"], "path_b": ["f2", "f3"]})
    groups = merge_duplicate_groups(table, pairs)
    stats = group_size_stats(table, groups, table["leaf_source"] == "fallback")
    assert stats == {
        "groups": 2,
        "images": 4,
        "mean": 2.0,
        "median": 2.0,
        "max": 3,
        "single_image_share": 0.5,
    }


def test_embeddings_roundtrip(tmp_path):
    emb = np.random.default_rng(0).normal(size=(3, 4)).astype(np.float32)
    save_embeddings(tmp_path / "e.npz", emb, ["a", "b", "c"])
    back = load_embeddings(tmp_path / "e.npz", ["a", "b", "c"])
    assert np.allclose(back, emb, atol=1e-2)
    assert load_embeddings(tmp_path / "e.npz", ["a", "c", "b"]) is None
    assert load_embeddings(tmp_path / "missing.npz", ["a"]) is None


def test_load_group_pairs_combines_files(tmp_path):
    assert load_group_pairs(tmp_path) is None
    pd.DataFrame({"path_a": ["x"], "path_b": ["y"], "corr_hi": [0.99]}).to_csv(
        tmp_path / "duplicate_pairs.csv", index=False
    )
    pd.DataFrame({"path_a": ["y", "p"], "path_b": ["z", "q"], "sim": [0.9, 0.8]}).to_csv(
        tmp_path / "sameleaf_pairs.csv", index=False
    )
    pairs = load_group_pairs(tmp_path)
    assert list(pairs.columns) == ["path_a", "path_b", "source"]
    assert pairs["source"].value_counts().to_dict() == {"sameleaf_pairs": 2, "duplicate_pairs": 1}


def test_embed_d4_is_pose_invariant(tmp_path):
    tf = pytest.importorskip("tensorflow")
    import keras
    from PIL import Image

    from plantdx.data.pipeline import make_dataset
    from plantdx.data.sameleaf import embed

    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    rows = []
    for name, arr in (
        ("orig", img),
        ("flip", img[:, ::-1]),
        ("other", rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)),
    ):
        path = tmp_path / "raw" / "color" / "X___y" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.ascontiguousarray(arr)).save(path)
        rows.append({"path": f"X___y/{name}.png", "label_index": 0})
    frame = pd.DataFrame(rows)
    keras.utils.set_random_seed(0)
    model = keras.Sequential(
        [
            keras.Input((16, 16, 3)),
            keras.layers.Rescaling(1 / 127.5, offset=-1),  # centre pixels: no brightness bias
            keras.layers.Flatten(),
            keras.layers.Dense(8, use_bias=False),
        ]
    )
    ds = make_dataset(frame, tmp_path, image_size=16, batch_size=2, num_classes=1)
    plain = embed(ds, model, tta="none")
    d4 = embed(ds, model, tta="d4")
    assert plain.shape == (3, 8) and np.allclose(np.linalg.norm(d4, axis=1), 1, atol=1e-5)
    assert plain[0] @ plain[1] < 0.99  # a flat dense layer sees the mirror as different
    assert d4[0] @ d4[1] == pytest.approx(1.0, abs=1e-4)  # D4 averaging removes the pose
    assert d4[0] @ d4[2] < 0.99
    with pytest.raises(ValueError):
        embed(ds, model, tta="rot")
    del tf
