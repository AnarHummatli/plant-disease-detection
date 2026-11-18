"""Build the leaf-grouped splits and write them to splits/ (plan steps 2.1-2.4).

    python -m plantdx.data.prepare

Downloads only the small index files (about 7 MB), not the 2.2 GB image archive, so it
runs on any machine with internet access. Outputs, all meant to be committed:

    splits/train.csv, val.csv, test.csv   path, label, label_index, leaf_id, leaf_source
    splits/classes.json                   the 38 class names; index = position in the list
    splits/dropped_train_overlap.csv      train rows removed because their leaf is in test
    splits/summary.json                   counts and the settings that produced the splits
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plantdx.data.source import (
    CLASSES,
    EXPECTED_SIZES,
    HF_REPO_ID,
    fetch_index_files,
    load_leaf_map,
    read_split_index,
)
from plantdx.data.split import (
    COLUMNS,
    SPLITS,
    assert_no_leakage,
    build_split_table,
    summarize,
)
from plantdx.paths import repo_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", default="color", choices=["color", "grayscale", "segmented"])
    parser.add_argument("--out", type=Path, default=repo_root() / "splits")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    files = fetch_index_files(args.variant)
    train = read_split_index(files["train"])
    test = read_split_index(files["test"])
    leaf_map = load_leaf_map(files["leaf_map"])

    expected = EXPECTED_SIZES.get(args.variant)
    if expected and (len(train), len(test)) != (expected["train"], expected["test"]):
        raise SystemExit(
            f"unexpected sizes: train {len(train)} / test {len(test)}, "
            f"dataset card says {expected['train']} / {expected['test']}"
        )
    seen = {r.label for r in train + test}
    if seen != set(CLASSES):
        raise SystemExit(f"class mismatch: {sorted(seen ^ set(CLASSES))}")

    table, dropped = build_split_table(train, test, leaf_map, args.val_fraction, args.seed)
    assert_no_leakage(table)

    args.out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        sub = table[table["split"] == split][COLUMNS]
        sub.to_csv(args.out / f"{split}.csv", index=False)
    dropped.to_csv(args.out / "dropped_train_overlap.csv", index=False)
    (args.out / "classes.json").write_text(
        json.dumps(list(CLASSES), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    summary = summarize(table, dropped)
    summary["settings"] = {
        "source": HF_REPO_ID,
        "variant": args.variant,
        "seed": args.seed,
        "val_fraction_requested": args.val_fraction,
        "official_train": len(train),
        "official_test": len(test),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nWritten to {args.out}. No leaf id is shared between train, val and test.")


if __name__ == "__main__":
    main()
