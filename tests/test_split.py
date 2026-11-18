"""Unit tests for the split builder, on small synthetic data."""

import pandas as pd
import pytest

from plantdx.data.source import CLASSES, parse_path, relative_path, zip_member
from plantdx.data.split import (
    assert_no_leakage,
    build_split_table,
    find_leakage,
    summarize,
)

CLS = list(CLASSES[:3])


def _rec(label, name):
    return parse_path(f"raw/color/{label}/{name}")


def make_data(leaves_per_class=8, imgs_per_leaf=3):
    """Mapped leaves 'L{c}_{k}' for train; test gets its own leaves. Returns records + map."""
    train, test, leaf_map = [], [], {}
    for c, label in enumerate(CLS):
        for k in range(leaves_per_class):
            for j in range(imgs_per_leaf):
                ident = f"src{c}_{k} {j}"
                train.append(_rec(label, f"u{c}{k}{j}___Src{c}_{k} {j}.JPG"))
                leaf_map[ident] = [f"{label}:::{c * 100 + k}.0"]
        for k in range(4):
            ident = f"tst{c}_{k} 1"
            test.append(_rec(label, f"t{c}{k}___Tst{c}_{k} 1.JPG"))
            leaf_map[ident] = [f"{label}:::{c * 100 + 50 + k}.0"]
    return train, test, leaf_map


def test_clean_data_has_no_leakage_and_all_classes_everywhere():
    train, test, leaf_map = make_data()
    table, dropped = build_split_table(train, test, leaf_map, val_fraction=0.25, seed=42)
    assert dropped.empty
    assert_no_leakage(table)
    for split in ("train", "val", "test"):
        assert set(table.loc[table["split"] == split, "label"]) == set(CLS)
    assert (table["split"] == "test").sum() == len(test)


def test_test_split_is_kept_as_is():
    train, test, leaf_map = make_data()
    table, _ = build_split_table(train, test, leaf_map, val_fraction=0.25)
    got = set(table.loc[table["split"] == "test", "path"])
    assert got == {relative_path(r.path) for r in test}


def test_overlapping_fallback_leaf_is_dropped_from_train_not_test():
    train, test, leaf_map = make_data()
    # same base name in train and test, no map entry -> identical fallback id
    train.append(_rec(CLS[0], "aaa___Dup_Leaf 7.JPG"))
    test.append(_rec(CLS[0], "bbb___Dup_Leaf 7.JPG"))
    table, dropped = build_split_table(train, test, leaf_map, val_fraction=0.25)
    assert len(dropped) == 1
    assert dropped.iloc[0]["leaf_id"] == "fallback_dup_leaf 7"
    assert "aaa___Dup_Leaf 7.JPG" not in " ".join(table.loc[table["split"] != "test", "path"])
    assert "bbb___Dup_Leaf 7.JPG" in " ".join(table.loc[table["split"] == "test", "path"])
    assert_no_leakage(table)


def test_leaf_source_flags():
    train, test, leaf_map = make_data()
    train.append(_rec(CLS[1], "zzz___No_Map 1.JPG"))
    table, _ = build_split_table(train, test, leaf_map, val_fraction=0.25)
    row = table[table["path"].str.contains("No_Map")].iloc[0]
    assert row["leaf_source"] == "fallback"
    assert (table[~table["path"].str.contains("No_Map")]["leaf_source"] == "map").all()


def test_images_of_one_leaf_stay_together():
    train, test, leaf_map = make_data(imgs_per_leaf=4)
    table, _ = build_split_table(train, test, leaf_map, val_fraction=0.25)
    per_leaf = table.groupby("leaf_id")["split"].nunique()
    assert (per_leaf == 1).all()


def test_split_is_deterministic_and_seed_dependent():
    train, test, leaf_map = make_data()
    a, _ = build_split_table(train, test, leaf_map, val_fraction=0.25, seed=42)
    b, _ = build_split_table(train, test, leaf_map, val_fraction=0.25, seed=42)
    c, _ = build_split_table(train, test, leaf_map, val_fraction=0.25, seed=7)
    pd.testing.assert_frame_equal(a, b)

    def val_paths(t):
        return set(t.loc[t["split"] == "val", "path"])

    assert val_paths(a) != val_paths(c)


def test_find_leakage_detects_shared_leaf():
    table = pd.DataFrame({"split": ["train", "val", "test"], "leaf_id": ["x", "x", "y"]})
    assert find_leakage(table) == {"train&val": {"x"}}
    with pytest.raises(AssertionError):
        assert_no_leakage(table)


def test_summarize_counts():
    train, test, leaf_map = make_data()
    table, dropped = build_split_table(train, test, leaf_map, val_fraction=0.25)
    s = summarize(table, dropped)
    assert s["train"]["images"] + s["val"]["images"] == len(train)
    assert s["test"]["images"] == len(test)
    assert s["dropped_train_images"] == 0


def test_relative_path_and_zip_member_roundtrip():
    member = "raw/color/Pepper,_bell___healthy/abc___IMG 1.JPG"
    rel = relative_path(member)
    assert rel == "Pepper,_bell___healthy/abc___IMG 1.JPG"
    assert zip_member("color", rel) == member


def test_csv_roundtrip_with_comma_in_class_name(tmp_path):
    rec = parse_path("raw/color/Pepper,_bell___healthy/abc___IMG 1.JPG")
    df = pd.DataFrame([{"path": relative_path(rec.path), "label": rec.label}])
    df.to_csv(tmp_path / "x.csv", index=False)
    back = pd.read_csv(tmp_path / "x.csv")
    assert back.loc[0, "label"] == "Pepper,_bell___healthy"
