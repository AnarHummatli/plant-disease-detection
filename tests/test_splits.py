"""Leakage tests on the COMMITTED split files (plan step 2.4).

They run after `python -m plantdx.data.prepare` has produced splits/*.csv. Until those
files exist the tests are skipped, so a fresh clone still passes.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from plantdx.data.source import CLASSES, EXPECTED_SIZES
from plantdx.data.split import assert_no_leakage

SPLITS_DIR = Path(__file__).resolve().parents[1] / "splits"
FILES = {s: SPLITS_DIR / f"{s}.csv" for s in ("train", "val", "test")}

pytestmark = pytest.mark.skipif(
    not all(f.exists() for f in FILES.values()),
    reason="splits/*.csv not generated yet (run: python -m plantdx.data.prepare)",
)


@pytest.fixture(scope="module")
def table():
    frames = [pd.read_csv(path).assign(split=name) for name, path in FILES.items()]
    return pd.concat(frames, ignore_index=True)


def test_no_leaf_group_or_duplicate_shared_between_splits(table):
    dup_file = SPLITS_DIR / "duplicate_pairs.csv"
    pairs = pd.read_csv(dup_file) if dup_file.exists() else None
    assert_no_leakage(table, pairs)


def test_group_column_present(table):
    assert "group_id" in table.columns
    # a group is a union of whole leaves
    assert (table.groupby("leaf_id")["group_id"].nunique() == 1).all()


def test_no_image_in_two_splits(table):
    assert not table["path"].duplicated().any()


def test_every_class_in_every_split(table):
    for split in ("train", "val", "test"):
        got = set(table.loc[table["split"] == split, "label_index"])
        assert got == set(range(len(CLASSES))), f"missing classes in {split}"


def test_label_index_matches_classes_json(table):
    classes = json.loads((SPLITS_DIR / "classes.json").read_text(encoding="utf-8"))
    assert classes == list(CLASSES)
    assert (table["label"] == table["label_index"].map(dict(enumerate(classes)))).all()


def test_test_split_is_official_size(table):
    assert (table["split"] == "test").sum() == EXPECTED_SIZES["color"]["test"]


def test_split_sizes_are_in_expected_range(table):
    n = len(table)
    share = table["split"].value_counts() / n
    assert 0.17 < share["test"] < 0.23
    assert 0.09 < share["val"] < 0.15
    assert 0.62 < share["train"] < 0.74
