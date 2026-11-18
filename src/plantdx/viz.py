"""Figures for the EDA step. Every function saves a PNG and returns its path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from plantdx.data.images import image_path  # noqa: E402


def _short(name: str) -> str:
    crop, _, disease = name.partition("___")
    return f"{crop.split('_(')[0]} · {disease}"


def plot_class_counts(table: pd.DataFrame, classes: list[str], out: str | Path) -> Path:
    counts = (
        table.groupby(["label_index", "split"]).size().unstack(fill_value=0).reindex(range(38))
    ).fillna(0)
    counts = counts[[c for c in ("train", "val", "test") if c in counts.columns]]
    fig, ax = plt.subplots(figsize=(10, 10))
    counts.plot.barh(stacked=True, ax=ax, width=0.8)
    ax.set_yticks(range(len(classes)), [_short(c) for c in classes], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("images")
    ax.set_title("Images per class and split")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return Path(out)


def plot_fallback_share(table: pd.DataFrame, classes: list[str], out: str | Path) -> Path:
    share = table.assign(fb=table["leaf_source"] == "fallback").groupby("label_index")["fb"].mean()
    share = share.reindex(range(len(classes))).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(range(len(classes)), share.to_numpy(), color="#c0504d")
    ax.set_yticks(range(len(classes)), [_short(c) for c in classes], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("% of images without a known leaf id (fallback)")
    ax.set_xlim(0, 100)
    ax.set_title("Where leaf grouping is unverified")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return Path(out)


def sample_grid(
    table: pd.DataFrame,
    classes: list[str],
    root: str | Path,
    variant: str,
    out: str | Path,
    cols: int = 8,
    seed: int = 42,
) -> Path:
    rows = (len(classes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.3))
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i >= len(classes):
            continue
        sub = table[table["label_index"] == i]
        if sub.empty:
            continue
        rel = sub.sample(1, random_state=seed + i).iloc[0]["path"]
        with Image.open(image_path(root, variant, rel)) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(_short(classes[i]), fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return Path(out)


def plot_audit_pairs(
    matches: pd.DataFrame, root: str | Path, variant: str, out: str | Path, n: int = 6
) -> Path:
    """Closest fallback test/val images next to their matches, for visual inspection."""
    pick = (
        matches[(matches["leaf_source"] == "fallback") & matches["match_path"].notna()]
        .sort_values("distance")
        .iloc[:: max(1, len(matches) // (n * 200))]
        .head(n)
    )
    fig, axes = plt.subplots(2, max(1, len(pick)), figsize=(max(1, len(pick)) * 2.4, 5.2))
    axes = axes.reshape(2, -1)
    for j, (_, row) in enumerate(pick.iterrows()):
        for k, (rel, tag) in enumerate(
            ((row["query_path"], "query"), (row["match_path"], "match"))
        ):
            with Image.open(image_path(root, variant, rel)) as im:
                axes[k, j].imshow(im.convert("RGB"))
            axes[k, j].axis("off")
            axes[k, j].set_title(f"{tag} d={row['distance']}" if k == 0 else tag, fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return Path(out)
