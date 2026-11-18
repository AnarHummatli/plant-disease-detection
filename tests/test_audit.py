"""Tests for image extraction, hashing and the cross-split near-duplicate search."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from plantdx.data.audit import (
    HASH_BITS,
    dhash,
    distance_quantiles,
    hamming_nearest,
    nearest_cross_split,
    positive_control,
    summarize_matches,
)
from plantdx.data.images import extract_variant, image_path, missing_files
from plantdx.viz import (
    plot_class_counts,
    plot_closest_pairs,
    plot_corr_hist,
    plot_fallback_share,
    plot_pairs,
    sample_grid,
)


def noise_image(seed, size=96):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)  # smooth-ish, low-frequency content
    return Image.fromarray(base).resize((size, size), Image.Resampling.BICUBIC)


def save_jpeg(img, path, quality=90):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=quality)
    return path


def popcount(a, b):
    return int(np.unpackbits((a ^ b).view(np.uint8)).sum())


def test_dhash_same_image_recompressed_is_close_and_different_is_far(tmp_path):
    img = noise_image(1)
    a = dhash(save_jpeg(img, tmp_path / "a.jpg", 95))
    a2 = dhash(save_jpeg(img.resize((80, 80)), tmp_path / "a2.jpg", 60))
    b = dhash(save_jpeg(noise_image(2), tmp_path / "b.jpg"))
    assert a.shape == (4,) and a.dtype == np.uint64
    assert popcount(a, a2) <= 30
    assert popcount(a, b) > 60
    assert popcount(a, a) == 0


def test_hamming_nearest_matches_bruteforce():
    rng = np.random.default_rng(0)
    pool = rng.integers(0, 2**63, (50, 4), dtype=np.uint64)
    queries = np.vstack([pool[7], rng.integers(0, 2**63, (3, 4), dtype=np.uint64)])
    d, idx = hamming_nearest(queries, pool, chunk=2)
    assert d[0] == 0 and idx[0] == 7
    for q, dist, i in zip(queries, d, idx, strict=True):
        expected = [popcount(q, p) for p in pool]
        assert dist == min(expected)
        assert expected[i] == min(expected)


def test_hamming_nearest_empty_pool():
    d, idx = hamming_nearest(np.zeros((2, 4), dtype=np.uint64), np.zeros((0, 4), dtype=np.uint64))
    assert (d == HASH_BITS + 1).all() and (idx == -1).all()


def test_nearest_cross_split_only_searches_same_class():
    rng = np.random.default_rng(3)
    h_a = rng.integers(0, 2**63, 4, dtype=np.uint64)
    table = pd.DataFrame(
        {
            "path": ["c0/train.jpg", "c1/train.jpg", "c0/test.jpg", "c1/test.jpg"],
            "label_index": [0, 1, 0, 1],
            "split": ["train", "train", "test", "test"],
            "leaf_source": ["map", "map", "fallback", "map"],
        }
    )
    hashes = np.stack(
        [h_a, rng.integers(0, 2**63, 4, dtype=np.uint64), h_a, h_a]  # c1/test equals c0/train
    )
    res = nearest_cross_split(table, hashes, "test", ("train", "val"))
    by_path = res.set_index("query_path")
    assert by_path.loc["c0/test.jpg", "distance"] == 0
    assert by_path.loc["c0/test.jpg", "match_path"] == "c0/train.jpg"
    assert by_path.loc["c1/test.jpg", "distance"] > 0  # must not match the class-0 image
    summary = summarize_matches(res)
    assert set(summary["leaf_source"]) == {"fallback", "map"}
    assert summary.set_index("leaf_source").loc["fallback", "<= 0"] == 100.0


def make_zip(tmp_path):
    zpath = tmp_path / "data.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for variant in ("color", "grayscale"):
            for i in range(3):
                buf = io.BytesIO()
                noise_image(i).save(buf, "JPEG")
                zf.writestr(f"raw/{variant}/Apple___healthy/img{i}.JPG", buf.getvalue())
    return zpath


def test_extract_variant_only_requested_variant_and_is_idempotent(tmp_path):
    zpath = make_zip(tmp_path)
    dest = tmp_path / "data"
    assert extract_variant(zpath, dest, "color") == 3
    assert extract_variant(zpath, dest, "color") == 0
    assert not (dest / "raw" / "grayscale").exists()
    rels = [f"Apple___healthy/img{i}.JPG" for i in range(3)]
    assert missing_files(dest, "color", rels) == []
    assert missing_files(dest, "color", [*rels, "Apple___healthy/nope.JPG"]) == [
        "Apple___healthy/nope.JPG"
    ]
    assert image_path(dest, "color", rels[0]).is_file()


def test_extract_variant_unknown_variant_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_variant(make_zip(tmp_path), tmp_path / "d", "segmented")


def test_viz_functions_write_pngs(tmp_path):
    classes = [f"Crop{i}___disease" for i in range(38)]
    rows = []
    for i in range(38):
        for s in ("train", "val", "test"):
            rel = f"{classes[i]}/{s}.JPG"
            save_jpeg(noise_image(i), tmp_path / "raw" / "color" / rel)
            rows.append(
                {
                    "path": rel,
                    "label_index": i,
                    "split": s,
                    "leaf_source": "fallback" if i % 2 else "map",
                }
            )
    table = pd.DataFrame(rows)
    for fn, args in (
        (plot_class_counts, (table, classes, tmp_path / "a.png")),
        (plot_fallback_share, (table, classes, tmp_path / "b.png")),
        (sample_grid, (table, classes, tmp_path, "color", tmp_path / "c.jpg")),
    ):
        assert fn(*args).stat().st_size > 1000
    with Image.open(tmp_path / "c.jpg") as im:
        assert im.format == "JPEG"


def test_plot_closest_pairs_without_threshold(tmp_path):
    for name in ("a", "b", "c", "d"):
        save_jpeg(noise_image(ord(name)), tmp_path / "raw" / "color" / "X___y" / f"{name}.jpg")
    pairs = pd.DataFrame(
        {
            "query_path": ["X___y/a.jpg", "X___y/b.jpg", "X___y/c.jpg"],
            "match_path": ["X___y/d.jpg", "X___y/d.jpg", None],
            "distance": [90, 70, 5],
        }
    )
    out = plot_closest_pairs(pairs, tmp_path, "color", tmp_path / "p.png", n=6)
    assert out.stat().st_size > 1000  # far pairs (70, 90) are still drawn
    with pytest.raises(ValueError):
        plot_closest_pairs(pairs.iloc[2:], tmp_path, "color", tmp_path / "q.png")


def test_positive_control_finds_planted_duplicate():
    rng = np.random.default_rng(5)
    dup = rng.integers(0, 2**63, 4, dtype=np.uint64)
    table = pd.DataFrame(
        {
            "path": ["c0/t1.jpg", "c0/t2.jpg", "c1/t3.jpg"],
            "label_index": [0, 0, 1],
            "split": ["test", "test", "test"],
            "leaf_source": ["fallback"] * 3,
        }
    )
    hashes = np.stack([rng.integers(0, 2**63, 4, dtype=np.uint64), dup, dup])
    dropped = pd.DataFrame(
        {
            "path": ["c0/d1.jpg", "c1/d2.jpg"],
            "label_index": [0, 1],
            "leaf_source": ["fallback", "fallback"],
        },
        index=[10, 11],  # misaligned index on purpose: must be reset internally
    )
    dropped_hashes = np.stack([dup, rng.integers(0, 2**63, 4, dtype=np.uint64)])
    res = positive_control(dropped, dropped_hashes, table, hashes).set_index("query_path")
    assert res.loc["c0/d1.jpg", "distance"] == 0
    assert res.loc["c0/d1.jpg", "match_path"] == "c0/t2.jpg"
    assert res.loc["c1/d2.jpg", "distance"] > 0  # c1's only test image is unrelated


def test_distance_quantiles_columns():
    m = pd.DataFrame(
        {"leaf_source": ["map"] * 3 + ["fallback"] * 2, "distance": [40, 80, 90, 30, 60]}
    )
    q = distance_quantiles(m).set_index("leaf_source")
    assert q.loc["fallback", "min"] == 30
    assert q.loc["map", "images"] == 3
    assert list(q.columns) == ["images", "min", "p01", "p05", "median", "max"]


def test_plot_pairs_applies_transform_and_hist(tmp_path):
    img = noise_image(9)
    save_jpeg(img, tmp_path / "raw" / "color" / "X___y" / "a.jpg")
    save_jpeg(
        img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        tmp_path / "raw" / "color" / "X___y" / "b.jpg",
    )
    pairs = pd.DataFrame(
        {"path_a": ["X___y/a.jpg"], "path_b": ["X___y/b.jpg"], "transform": ["hflip"]}
    )
    out = plot_pairs(pairs, tmp_path, "color", tmp_path / "p.jpg", labels=["r=0.99"])
    assert out.stat().st_size > 1000
    cand = pd.DataFrame(
        {"corr_hi": [0.91, 0.95, 0.99, 1.0], "same_leaf": [True, False, False, True]}
    )
    assert plot_corr_hist(cand, 0.97, tmp_path / "h.png").stat().st_size > 1000
