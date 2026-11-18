"""Generated parts of docs/data_card.md (plan step 2.6).

    python -m plantdx.data.card          # rewrite the generated block in docs/data_card.md
    python -m plantdx.data.card --check  # exit 1 if the block is out of date

The class table is built from splits/*.csv, so the data card cannot drift away from the data;
tests/test_data_card.py runs the --check logic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from plantdx.paths import repo_root

START = "<!-- generated:class-table:start -->"
END = "<!-- generated:class-table:end -->"
SPLITS = ("train", "val", "test")


def load_tables(splits_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.concat(
        [pd.read_csv(splits_dir / f"{s}.csv").assign(split=s) for s in SPLITS], ignore_index=True
    )
    dropped = pd.read_csv(splits_dir / "dropped_train_overlap.csv")
    return table, dropped


def class_table(table: pd.DataFrame, dropped: pd.DataFrame) -> pd.DataFrame:
    """One row per class: split counts, removed images, share without a verified leaf id."""
    counts = table.groupby(["label_index", "split"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=list(SPLITS), fill_value=0)
    labels = table.drop_duplicates("label_index").set_index("label_index")["label"]
    removed = dropped.groupby("label_index").size()
    everything = pd.concat([table, dropped], ignore_index=True)
    fallback = everything.groupby("label_index")["leaf_source"].apply(
        lambda s: (s == "fallback").mean()
    )
    out = pd.DataFrame(
        {
            "label": labels,
            "train": counts["train"],
            "val": counts["val"],
            "test": counts["test"],
            "removed": removed.reindex(counts.index, fill_value=0),
            "fallback_pct": (fallback * 100).round(0).astype(int),
        }
    ).sort_index()
    out["total"] = out[["train", "val", "test", "removed"]].sum(axis=1)
    return out


def render(ct: pd.DataFrame) -> str:
    lines = [
        START,
        "<!-- Do not edit by hand: `python -m plantdx.data.card` rewrites this block. -->",
        "",
        "| # | Crop | Condition | Train | Val | Test | Removed | Total | No leaf id |",
        "|--:|---|---|--:|--:|--:|--:|--:|--:|",
    ]
    for idx, r in ct.iterrows():
        crop, _, cond = r["label"].partition("___")
        crop = crop.replace("_", " ").replace(",", "")
        cond = cond.replace("_", " ").strip()
        lines.append(
            f"| {idx} | {crop} | {cond} | {r.train:,} | {r.val:,} | {r.test:,} | {r.removed:,} "
            f"| {r.total:,} | {r.fallback_pct}% |"
        )
    tot = ct[["train", "val", "test", "removed", "total"]].sum()
    fb_all = round((ct["fallback_pct"] * ct["total"]).sum() / tot["total"])
    lines.append(
        f"| | **All 38 classes** | | **{tot.train:,}** | **{tot.val:,}** | **{tot.test:,}** "
        f"| **{tot.removed:,}** | **{tot.total:,}** | **{fb_all}%** |"
    )
    lines += ["", END]
    return "\n".join(lines)


def replace_block(text: str, block: str) -> str:
    if START not in text or END not in text:
        raise ValueError("data card has no generated class-table block")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    return head + block + tail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--card", type=Path, default=repo_root() / "docs" / "data_card.md")
    parser.add_argument("--splits", type=Path, default=repo_root() / "splits")
    parser.add_argument("--check", action="store_true", help="only check, do not write")
    args = parser.parse_args(argv)

    block = render(class_table(*load_tables(args.splits)))
    text = args.card.read_text(encoding="utf-8")
    new = replace_block(text, block)
    if args.check:
        if new != text:
            print("docs/data_card.md is out of date: run python -m plantdx.data.card")
            return 1
        print("data card is up to date")
        return 0
    args.card.write_text(new, encoding="utf-8")
    print(f"updated {args.card}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
