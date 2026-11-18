"""Tests for the copy-of-the-same-photo search (plantdx.data.duplicates)."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageFilter

from plantdx.data.duplicates import (
    D4,
    annotate,
    apply_transform,
    candidate_pairs,
    load_gray,
    normalize,
    summarize_duplicates,
    thumbnails,
    verify_pairs,
)


def leaf_like(seed, size=128):
    """Smooth random image with fine texture: different seeds give unrelated images."""
    rng = np.random.default_rng(seed)
    coarse = Image.fromarray(rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)).resize(
        (size, size), Image.Resampling.BICUBIC
    )
    fine = rng.integers(-25, 25, (size, size, 3))
    return Image.fromarray(np.clip(np.asarray(coarse, int) + fine, 0, 255).astype(np.uint8))


@pytest.mark.parametrize("name", D4)
def test_transforms_are_bijections_matching_numpy(name):
    x = np.arange(16).reshape(4, 4)
    y = apply_transform(x, name)
    assert sorted(y.ravel()) == list(range(16))
    if name == "rot90":
        assert (y == np.rot90(x)).all()
    if name == "transpose":
        assert (y == x.T).all()


def test_transform_inverse_pairs():
    x = np.arange(9).reshape(3, 3)
    assert (apply_transform(apply_transform(x, "rot90"), "rot270") == x).all()
    for t in ("hflip", "vflip", "rot180", "transpose", "antitranspose"):
        assert (apply_transform(apply_transform(x, t), t) == x).all()
    with pytest.raises(ValueError):
        apply_transform(x, "shear")


def test_normalize_gives_correlation_and_handles_flat_images():
    a = np.random.default_rng(0).integers(0, 256, (2, 8, 8)).astype(np.uint8)
    flat = np.full((1, 8, 8), 7, dtype=np.uint8)
    n = normalize(np.concatenate([a, flat]))
    assert np.allclose(np.linalg.norm(n[:2], axis=1), 1, atol=1e-5)
    assert (n[2] == 0).all()
    expected = np.corrcoef(a[0].ravel(), a[1].ravel())[0, 1]
    assert np.isclose(n[0] @ n[1], expected, atol=1e-5)


@pytest.fixture
def dataset(tmp_path):
    """Class 0: original, blurred+recompressed copy, mirrored copy, rotated copy, two others.
    Class 1: an exact copy of class-0's original (must NOT be paired across classes)."""
    root = tmp_path / "raw" / "color"
    orig = leaf_like(1)
    files = {
        "A/orig.jpg": orig,
        "A/blurred.jpg": orig.filter(ImageFilter.GaussianBlur(1.2)).resize((112, 112)),
        "A/mirror.jpg": orig.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        "A/rotated.jpg": orig.transpose(Image.Transpose.ROTATE_90),
        "A/other1.jpg": leaf_like(2),
        "A/other2.jpg": leaf_like(3),
        "B/copy_other_class.jpg": orig,
    }
    for rel, img in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        img.save(root / rel, "JPEG", quality=70 if "blurred" in rel else 92)
    table = pd.DataFrame(
        {
            "path": list(files),
            "label_index": [0] * 6 + [1],
            "split": ["train", "test", "val", "train", "train", "test", "train"],
            "leaf_id": ["L1", "L2", "L3", "L4", "L5", "L6", "L7"],
            "leaf_source": ["map", "fallback", "map", "map", "map", "map", "map"],
        }
    )
    return root, table


def find(root, table, min_corr=0.9):
    thumbs = thumbnails([root / p for p in table["path"]], size=32, workers=2)
    cand = candidate_pairs(table, thumbs, min_corr=min_corr, chunk=2)
    return verify_pairs(cand, lambda p, s: load_gray(root / p, s), size=96, workers=2)


def test_finds_blurred_mirrored_and_rotated_copies(dataset):
    root, table = dataset
    pairs = find(root, table)
    got = {frozenset((a, b)): t for a, b, t in pairs[["path_a", "path_b", "transform"]].values}
    assert frozenset(("A/orig.jpg", "A/blurred.jpg")) in got
    assert got[frozenset(("A/orig.jpg", "A/mirror.jpg"))] == "hflip"
    assert got[frozenset(("A/orig.jpg", "A/rotated.jpg"))] in ("rot90", "rot270")
    copies = pairs[pairs.apply(lambda r: "other" not in r.path_a + r.path_b, axis=1)]
    assert (copies["corr_hi"] > 0.95).all()
    # unrelated images never reach the candidate list
    assert not pairs["path_a"].str.contains("other").any()
    assert not pairs["path_b"].str.contains("other").any()


def test_never_pairs_across_classes(dataset):
    root, table = dataset
    pairs = find(root, table)
    assert not (pairs["path_a"].str.startswith("B/") | pairs["path_b"].str.startswith("B/")).any()


def test_chunking_does_not_change_result(dataset):
    root, table = dataset
    thumbs = thumbnails([root / p for p in table["path"]], size=32, workers=1)
    a = candidate_pairs(table, thumbs, min_corr=0.9, chunk=1)
    b = candidate_pairs(table, thumbs, min_corr=0.9, chunk=1000)
    key = ["path_a", "path_b"]
    pd.testing.assert_frame_equal(
        a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
    )


def test_max_pairs_guard(dataset):
    root, table = dataset
    thumbs = thumbnails([root / p for p in table["path"]], size=32, workers=1)
    with pytest.raises(RuntimeError):
        candidate_pairs(table, thumbs, min_corr=-1.0, max_pairs=3)


def test_empty_candidates_and_verify():
    table = pd.DataFrame({"path": ["x"], "label_index": [0]})
    empty = candidate_pairs(table, np.zeros((1, 8, 8), np.uint8))
    assert empty.empty and "corr32" in empty.columns
    assert "corr_hi" in verify_pairs(empty, load=None).columns


def test_annotate_and_summarize(dataset):
    root, table = dataset
    pairs = annotate(find(root, table), table)
    assert {"cross_split", "same_leaf", "split_a", "split_b"} <= set(pairs.columns)
    summary = summarize_duplicates(pairs[pairs["cross_split"]]).set_index("splits")
    assert summary.loc["test&train", "pairs"] >= 1  # orig (train) vs blurred (test)
    assert summary.loc["test&train", "fallback_involved"] >= 1
    assert summarize_duplicates(pairs.iloc[0:0]).empty
