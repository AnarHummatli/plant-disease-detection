"""The data card's class table must match the committed splits (plan 2.6)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from plantdx.data.card import END, START, class_table, load_tables, main, render, replace_block

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "splits"
CARD = ROOT / "docs" / "data_card.md"

needs_splits = pytest.mark.skipif(
    not (SPLITS / "train.csv").exists(), reason="splits not generated yet"
)


@needs_splits
def test_data_card_is_up_to_date():
    assert main(["--check", "--card", str(CARD), "--splits", str(SPLITS)]) == 0


@needs_splits
def test_class_table_totals_match_summary():
    ct = class_table(*load_tables(SPLITS))
    summary = json.loads((SPLITS / "summary.json").read_text())
    assert len(ct) == 38
    for split in ("train", "val", "test"):
        assert ct[split].sum() == summary[split]["images"]
    assert ct["removed"].sum() == summary["dropped_train_images"]
    assert (
        ct["total"].sum()
        == summary["settings"]["official_train"] + summary["settings"]["official_test"]
    )


def test_render_and_replace_block():
    table = pd.DataFrame(
        {
            "label": ["A___x", "A___x", "B___healthy"],
            "label_index": [0, 0, 1],
            "split": ["train", "test", "val"],
            "leaf_source": ["map", "fallback", "map"],
        }
    )
    dropped = pd.DataFrame({"label_index": [1], "leaf_source": ["fallback"]})
    ct = class_table(table, dropped)
    assert ct.loc[0, ["train", "val", "test", "removed", "total"]].tolist() == [1, 0, 1, 0, 2]
    assert ct.loc[1, "fallback_pct"] == 50
    block = render(ct)
    assert block.startswith(START) and block.endswith(END)
    text = f"intro\n{START}\nold\n{END}\noutro"
    new = replace_block(text, block)
    assert new.startswith("intro\n") and new.endswith("\noutro") and "old" not in new
    with pytest.raises(ValueError):
        replace_block("no markers", block)
