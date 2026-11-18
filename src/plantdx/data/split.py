"""Leaf-grouped train / val / test table (plan steps 2.3-2.4).

Rules
-----
1. The official test split is kept exactly as released.
2. Official-train images whose leaf id also occurs in the official test are removed from
   train. Measured on the real data: 227 leaf ids, all of them "fallback" ids (image names
   with no entry in leaf-map.json), covering 290 train and 243 test images. See D-004.
3. Validation is carved out of the remaining train images with StratifiedGroupKFold,
   grouped by leaf id, so a leaf never straddles train and val.
4. Every row keeps `leaf_source`: "map" (leaf id taken from leaf-map.json) or "fallback"
   (leaf id guessed from the file name). Metrics can then be reported on the verified
   "map" subset separately.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from plantdx.data.source import (
    Record,
    leaf_identifier,
    relative_path,
    resolve_leaf_id,
)

COLUMNS = ["path", "label", "label_index", "leaf_id", "leaf_source"]
SPLITS = ("train", "val", "test")


def records_to_frame(records: list[Record], leaf_map: dict[str, list[str]]) -> pd.DataFrame:
    """One row per image with its resolved leaf id."""
    rows = []
    for rec in records:
        leaf = resolve_leaf_id(leaf_identifier(rec.path), rec.label, leaf_map)
        rows.append(
            {
                "path": relative_path(rec.path),
                "label": rec.label,
                "label_index": rec.label_index,
                "leaf_id": leaf,
                "leaf_source": "fallback" if leaf.startswith("fallback_") else "map",
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def build_split_table(
    train_records: list[Record],
    test_records: list[Record],
    leaf_map: dict[str, list[str]],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (table, dropped).

    `table` has COLUMNS + "split" (train / val / test); `dropped` holds the official-train
    rows removed because their leaf id also occurs in the official test split.
    """
    official_train = records_to_frame(train_records, leaf_map)
    test = records_to_frame(test_records, leaf_map)

    overlap = set(official_train["leaf_id"]) & set(test["leaf_id"])
    is_overlap = official_train["leaf_id"].isin(overlap)
    dropped = official_train[is_overlap].reset_index(drop=True)
    kept = official_train[~is_overlap].reset_index(drop=True)

    n_splits = max(2, round(1 / val_fraction))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(iter(cv.split(kept, kept["label_index"], groups=kept["leaf_id"])))

    parts = [
        kept.iloc[train_idx].assign(split="train"),
        kept.iloc[val_idx].assign(split="val"),
        test.assign(split="test"),
    ]
    table = pd.concat(parts, ignore_index=True)
    table = table.sort_values(["split", "path"], kind="stable").reset_index(drop=True)
    return table, dropped


def find_leakage(table: pd.DataFrame) -> dict[str, set[str]]:
    """Leaf ids that occur in more than one split, per split pair ('train&val', ...)."""
    ids = {s: set(table.loc[table["split"] == s, "leaf_id"]) for s in SPLITS}
    return {
        f"{a}&{b}": ids[a] & ids[b]
        for i, a in enumerate(SPLITS)
        for b in SPLITS[i + 1 :]
        if ids[a] & ids[b]
    }


def assert_no_leakage(table: pd.DataFrame) -> None:
    leaks = find_leakage(table)
    if leaks:
        detail = ", ".join(f"{pair}: {len(ids)} leaves" for pair, ids in leaks.items())
        raise AssertionError(f"leaf ids shared between splits ({detail})")


def summarize(table: pd.DataFrame, dropped: pd.DataFrame) -> dict:
    """Counts for splits/summary.json and the data card."""
    out: dict = {"dropped_train_images": int(len(dropped))}
    out["dropped_leaf_ids"] = int(dropped["leaf_id"].nunique())
    for split in SPLITS:
        sub = table[table["split"] == split]
        out[split] = {
            "images": int(len(sub)),
            "leaves": int(sub["leaf_id"].nunique()),
            "classes": int(sub["label_index"].nunique()),
            "map_images": int((sub["leaf_source"] == "map").sum()),
            "fallback_images": int((sub["leaf_source"] == "fallback").sum()),
        }
    return out
