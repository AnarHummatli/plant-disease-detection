"""Leaf-grouped train / val / test table (plan steps 2.3-2.4).

Rules
-----
1. The official test split is kept exactly as released.
2. Images are grouped. A group is one physical leaf (`leaf_id`), extended with every image
   that is a copy of the same photo (`duplicate_pairs`, found by plantdx.data.duplicates):
   copies are joined into one group with union-find, so they can never be on two sides.
3. Official-train images whose group also occurs in the official test are removed from
   train (listed in splits/dropped_train_overlap.csv). See D-004 and D-006.
4. Validation is carved out of the remaining train images with StratifiedGroupKFold,
   grouped by `group_id`, so a group never straddles train and val.
5. Every row keeps `leaf_source`: "map" (leaf id taken from leaf-map.json) or "fallback"
   (leaf id guessed from the file name), so metrics can be reported separately for the
   leaf-verified subset.
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

COLUMNS = ["path", "label", "label_index", "leaf_id", "group_id", "leaf_source"]
SPLITS = ("train", "val", "test")
GROUP_KEYS = ("leaf_id", "group_id")


def records_to_frame(records: list[Record], leaf_map: dict[str, list[str]]) -> pd.DataFrame:
    """One row per image with its resolved leaf id (group_id = leaf_id until merged)."""
    rows = []
    for rec in records:
        leaf = resolve_leaf_id(leaf_identifier(rec.path), rec.label, leaf_map)
        rows.append(
            {
                "path": relative_path(rec.path),
                "label": rec.label,
                "label_index": rec.label_index,
                "leaf_id": leaf,
                "group_id": leaf,
                "leaf_source": "fallback" if leaf.startswith("fallback_") else "map",
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # the smaller id becomes the root: deterministic group names
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


def merge_duplicate_groups(frame: pd.DataFrame, duplicate_pairs: pd.DataFrame | None) -> pd.Series:
    """group_id for every row of `frame`: its leaf id, merged across duplicate pairs.

    Pairs that mention a path not in `frame` are ignored. Returns a Series aligned to `frame`.
    """
    if duplicate_pairs is None or duplicate_pairs.empty:
        return frame["leaf_id"].copy()
    leaf_of = dict(zip(frame["path"], frame["leaf_id"], strict=True))
    uf = _UnionFind()
    for a, b in zip(duplicate_pairs["path_a"], duplicate_pairs["path_b"], strict=True):
        if a in leaf_of and b in leaf_of:
            uf.union(leaf_of[a], leaf_of[b])
    return frame["leaf_id"].map(uf.find)


def build_split_table(
    train_records: list[Record],
    test_records: list[Record],
    leaf_map: dict[str, list[str]],
    val_fraction: float = 0.15,
    seed: int = 42,
    duplicate_pairs: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (table, dropped).

    `table` has COLUMNS + "split" (train / val / test); `dropped` holds the official-train
    rows removed because their group also occurs in the official test split.
    """
    official_train = records_to_frame(train_records, leaf_map).assign(_official="train")
    test = records_to_frame(test_records, leaf_map).assign(_official="test")
    everything = pd.concat([official_train, test], ignore_index=True)
    everything["group_id"] = merge_duplicate_groups(everything, duplicate_pairs)

    official_train = everything[everything["_official"] == "train"].drop(columns="_official")
    test = everything[everything["_official"] == "test"].drop(columns="_official")

    overlap = set(official_train["group_id"]) & set(test["group_id"])
    is_overlap = official_train["group_id"].isin(overlap)
    dropped = official_train[is_overlap].reset_index(drop=True)
    kept = official_train[~is_overlap].reset_index(drop=True)

    n_splits = max(2, round(1 / val_fraction))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(iter(cv.split(kept, kept["label_index"], groups=kept["group_id"])))

    parts = [
        kept.iloc[train_idx].assign(split="train"),
        kept.iloc[val_idx].assign(split="val"),
        test.assign(split="test"),
    ]
    table = pd.concat(parts, ignore_index=True)
    table = table.sort_values(["split", "path"], kind="stable").reset_index(drop=True)
    return table, dropped


def find_leakage(table: pd.DataFrame, key: str = "leaf_id") -> dict[str, set[str]]:
    """Values of `key` that occur in more than one split, per split pair ('train&val', ...)."""
    ids = {s: set(table.loc[table["split"] == s, key]) for s in SPLITS}
    return {
        f"{a}&{b}": ids[a] & ids[b]
        for i, a in enumerate(SPLITS)
        for b in SPLITS[i + 1 :]
        if ids[a] & ids[b]
    }


def cross_split_pairs(table: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """Rows of `pairs` (path_a, path_b) whose two images are in different splits.

    Images that are not in `table` (e.g. removed train images) are ignored.
    """
    split_of = dict(zip(table["path"], table["split"], strict=True))
    a = pairs["path_a"].map(split_of)
    b = pairs["path_b"].map(split_of)
    return pairs[a.notna() & b.notna() & (a != b)]


def assert_no_leakage(table: pd.DataFrame, duplicate_pairs: pd.DataFrame | None = None) -> None:
    """Fail if a leaf, a duplicate group or a duplicate pair crosses splits."""
    problems = []
    for key in GROUP_KEYS:
        if key in table.columns:
            for pair, ids in find_leakage(table, key).items():
                problems.append(f"{key} {pair}: {len(ids)}")
    if duplicate_pairs is not None and not duplicate_pairs.empty:
        n = len(cross_split_pairs(table, duplicate_pairs))
        if n:
            problems.append(f"duplicate pairs across splits: {n}")
    if problems:
        raise AssertionError("leakage between splits (" + "; ".join(problems) + ")")


def summarize(table: pd.DataFrame, dropped: pd.DataFrame) -> dict:
    """Counts for splits/summary.json and the data card."""
    out: dict = {
        "dropped_train_images": int(len(dropped)),
        "dropped_leaf_ids": int(dropped["leaf_id"].nunique()),
        "dropped_groups": int(dropped["group_id"].nunique()),
        "dropped_by_leaf_source": {
            k: int(v) for k, v in dropped["leaf_source"].value_counts().sort_index().items()
        },
        "merged_groups": int((table.groupby("group_id")["leaf_id"].nunique() > 1).sum()),
    }
    for split in SPLITS:
        sub = table[table["split"] == split]
        out[split] = {
            "images": int(len(sub)),
            "leaves": int(sub["leaf_id"].nunique()),
            "groups": int(sub["group_id"].nunique()),
            "classes": int(sub["label_index"].nunique()),
            "map_images": int((sub["leaf_source"] == "map").sum()),
            "fallback_images": int((sub["leaf_source"] == "fallback").sum()),
        }
    return out


# Pair files that join images into one group before splitting, in splits/ (D-006, D-010).
PAIR_FILES = ("duplicate_pairs.csv", "sameleaf_pairs.csv")


def load_group_pairs(splits_dir, names=PAIR_FILES) -> pd.DataFrame | None:
    """All grouping pairs in `splits_dir` (path_a, path_b, source), or None if there are none."""
    from pathlib import Path

    frames = []
    for name in names:
        f = Path(splits_dir) / name
        if f.exists():
            frames.append(pd.read_csv(f)[["path_a", "path_b"]].assign(source=f.stem))
    return pd.concat(frames, ignore_index=True) if frames else None
