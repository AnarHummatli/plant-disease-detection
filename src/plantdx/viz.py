"""Figures for the EDA step. Every function saves a PNG and returns its path."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from plantdx.data.images import image_path  # noqa: E402


def _save(fig, out: str | Path, dpi: int) -> Path:
    """Save a figure; .jpg/.jpeg outputs are written as JPEG (much smaller for photo grids)."""
    out = Path(out)
    kwargs = {}
    if out.suffix.lower() in (".jpg", ".jpeg"):
        kwargs["pil_kwargs"] = {"quality": 85, "optimize": True}
    fig.savefig(out, dpi=dpi, **kwargs)
    plt.close(fig)
    return out


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
    return _save(fig, out, dpi=150)


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
    return _save(fig, out, dpi=150)


def sample_grid(
    table: pd.DataFrame,
    classes: list[str],
    root: str | Path,
    variant: str,
    out: str | Path,
    cols: int = 8,
    seed: int = 42,
    dpi: int = 110,
) -> Path:
    """One random image per class. Save as .jpg: a PNG of 38 photos is several MB."""
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
    return _save(fig, out, dpi=dpi)


def plot_closest_pairs(
    pairs: pd.DataFrame,
    root: str | Path,
    variant: str,
    out: str | Path,
    n: int = 6,
    title: str = "",
) -> Path:
    """The `n` pairs with the smallest hash distance: query image above, its match below.

    `pairs` needs columns query_path, match_path, distance. No distance threshold is applied,
    so the figure always shows the closest pairs that exist, however far apart they are.
    """
    pick = pairs[pairs["match_path"].notna()].nsmallest(n, "distance")
    if pick.empty:
        raise ValueError("no pairs with a match to plot")
    cols = len(pick)
    fig, axes = plt.subplots(2, cols, figsize=(cols * 2.2, 5.0), squeeze=False)
    for j, (_, row) in enumerate(pick.iterrows()):
        for k, rel in enumerate((row["query_path"], row["match_path"])):
            with Image.open(image_path(root, variant, rel)) as im:
                axes[k, j].imshow(im.convert("RGB"))
            axes[k, j].set_xticks([])
            axes[k, j].set_yticks([])
        axes[0, j].set_title(f"distance {int(row['distance'])}", fontsize=8)
        axes[1, j].set_xlabel(_short(row["query_path"].split("/", 1)[0]), fontsize=6)
    axes[0, 0].set_ylabel("query", fontsize=8)
    axes[1, 0].set_ylabel("nearest match", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return _save(fig, out, dpi=120)
