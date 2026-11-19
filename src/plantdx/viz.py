"""Figures for the report (EDA, training, error analysis). Every function saves an image and
returns its path; photo grids are saved as JPEG."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
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
    ax.set_ylabel("")
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


def _rgb(root: str | Path, variant: str, rel: str, transform: str | None = None) -> np.ndarray:
    with Image.open(image_path(root, variant, rel)) as im:
        arr = np.asarray(im.convert("RGB"))
    if transform and transform != "identity":
        from plantdx.data.duplicates import apply_transform

        arr = np.moveaxis(apply_transform(np.moveaxis(arr, -1, 0), transform), 0, -1)
    return arr


def plot_pairs(
    pairs: pd.DataFrame,
    root: str | Path,
    variant: str,
    out: str | Path,
    labels: list[str] | None = None,
    title: str = "",
) -> Path:
    """Draw the given pairs (in order): image a above, image b below.

    If `pairs` has a `transform` column, image b is shown after applying it, so both images
    appear in the same orientation. `labels` are the column titles.
    """
    pairs = pairs[pairs["path_b"].notna()]
    if pairs.empty:
        raise ValueError("no pairs to plot")
    cols = len(pairs)
    fig, axes = plt.subplots(2, cols, figsize=(cols * 2.2, 5.0), squeeze=False)
    transforms = pairs["transform"] if "transform" in pairs else [None] * cols
    for j, ((_, row), t) in enumerate(zip(pairs.iterrows(), transforms, strict=True)):
        axes[0, j].imshow(_rgb(root, variant, row["path_a"]))
        axes[1, j].imshow(_rgb(root, variant, row["path_b"], t))
        for k in (0, 1):
            axes[k, j].set_xticks([])
            axes[k, j].set_yticks([])
        if labels:
            axes[0, j].set_title(labels[j], fontsize=7)
        axes[1, j].set_xlabel(_short(row["path_a"].split("/", 1)[0]), fontsize=6)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return _save(fig, out, dpi=110)


def plot_closest_pairs(
    pairs: pd.DataFrame,
    root: str | Path,
    variant: str,
    out: str | Path,
    n: int = 6,
    title: str = "",
) -> Path:
    """The `n` pairs with the smallest hash distance (columns query_path, match_path, distance).

    No distance threshold is applied, so the figure always shows the closest pairs that exist.
    """
    pick = pairs[pairs["match_path"].notna()].nsmallest(n, "distance")
    if pick.empty:
        raise ValueError("no pairs with a match to plot")
    renamed = pick.rename(columns={"query_path": "path_a", "match_path": "path_b"})
    labels = [f"distance {int(d)}" for d in pick["distance"]]
    return plot_pairs(renamed, root, variant, out, labels=labels, title=title)


def plot_corr_hist(pairs: pd.DataFrame, threshold: float, out: str | Path) -> Path:
    """Histogram of stage-2 correlations of candidate pairs, with the chosen cut-off."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    groups = (
        ("different leaf id", pairs[~pairs["same_leaf"]]),
        ("same leaf id", pairs[pairs["same_leaf"]]),
    )
    bins = np.linspace(pairs["corr_hi"].min() if len(pairs) else 0, 1, 60)
    for name, g in groups:
        ax.hist(g["corr_hi"], bins=bins, alpha=0.6, label=f"{name} ({len(g):,})", log=True)
    ax.axvline(threshold, color="k", ls="--", lw=1, label=f"cut-off {threshold}")
    ax.set_xlabel("correlation at 96x96 (best D4 transform)")
    ax.set_ylabel("pairs (log)")
    ax.legend(fontsize=7)
    ax.set_title("Candidate pairs from stage 1")
    fig.tight_layout()
    return _save(fig, out, dpi=130)


def plot_augmentations(
    frame: pd.DataFrame,
    root: str | Path,
    variant: str,
    out: str | Path,
    augment=None,
    image_size: int = 224,
    n_images: int = 4,
    n_aug: int = 5,
    seed: int = 42,
) -> Path:
    """Each row: one image as the model sees it at evaluation, then `n_aug` training versions."""
    import tensorflow as tf

    from plantdx.data.pipeline import build_augmentation

    aug = build_augmentation(augment, seed=seed)
    pick = frame.sample(min(n_images, len(frame)), random_state=seed)
    fig, axes = plt.subplots(
        len(pick), n_aug + 1, figsize=((n_aug + 1) * 1.9, len(pick) * 2.0), squeeze=False
    )
    for r, rel in enumerate(pick["path"]):
        with Image.open(image_path(root, variant, rel)) as im:
            base = im.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        x = tf.repeat(tf.constant(np.asarray(base, np.float32))[None], n_aug, axis=0)
        versions = [np.asarray(base)] + [
            np.clip(v, 0, 255).astype(np.uint8)
            for v in (aug(x, training=True).numpy() if aug.layers else x.numpy())
        ]
        for c, img in enumerate(versions):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title("original" if c == 0 else f"train #{c}", fontsize=8)
        axes[r, 0].set_ylabel(_short(rel.split("/", 1)[0]), fontsize=6)
    fig.suptitle("Training augmentation (flip, rotation, zoom, brightness, contrast)", fontsize=9)
    fig.tight_layout()
    return _save(fig, out, dpi=100)


def plot_images_per_leaf(table: pd.DataFrame, out: str | Path) -> Path:
    """How many images each leaf id has, verified (map) vs file-name (fallback) ids."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    sizes = {
        s: table[table["leaf_source"] == s].groupby("leaf_id").size() for s in ("map", "fallback")
    }
    top = int(max(v.max() for v in sizes.values() if len(v)))
    bins = np.arange(1, top + 2) - 0.5
    for name, color in (("map", "#2E7D32"), ("fallback", "#c0504d")):
        v = sizes[name]
        kind = "verified" if name == "map" else "file-name"
        label = f"{kind} leaf ids ({len(v):,}; median {v.median():g})"
        ax.hist(v, bins=bins, alpha=0.65, color=color, label=label, log=True)
    ax.set_xlabel("images per leaf id")
    ax.set_ylabel("leaf ids (log)")
    ax.set_title("Images per leaf: verified leaves have ~4 photos, file-name ids almost always 1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out, dpi=130)


def plot_learning_curves(history_csv: str | Path, out: str | Path, title: str = "") -> Path:
    """Loss, accuracy and macro-F1 per epoch, train vs val, from a Keras CSVLogger file."""
    hist = pd.read_csv(history_csv)
    epochs = hist["epoch"].to_numpy() + 1 if "epoch" in hist else np.arange(1, len(hist) + 1)
    panels = [(m, m.replace("_", "-")) for m in ("loss", "accuracy", "macro_f1") if m in hist]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4), squeeze=False)
    for ax, (metric, label) in zip(axes[0], panels, strict=True):
        ax.plot(epochs, hist[metric], marker=".", label="train")
        if f"val_{metric}" in hist:
            ax.plot(epochs, hist[f"val_{metric}"], marker=".", label="val")
        ax.set_xlabel("epoch")
        ax.set_title(label)
        ax.grid(alpha=0.3)
    if "val_macro_f1" in hist:
        best = int(hist["val_macro_f1"].to_numpy().argmax())
        for ax in axes[0]:
            ax.axvline(epochs[best], color="k", ls=":", lw=1)
        axes[0][-1].annotate(
            f"best {hist['val_macro_f1'].iloc[best]:.4f}",
            (epochs[best], hist["val_macro_f1"].iloc[best]),
            textcoords="offset points",
            xytext=(-60, -14),
            fontsize=8,
        )
    axes[0][0].legend(fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return _save(fig, out, dpi=130)


# ------------------------------------------------------------------ error analysis (stage 5)
def plot_confusion(
    cm_norm: np.ndarray, classes: list[str], out: str | Path, title: str = ""
) -> Path:
    """Row-normalised confusion matrix (recall on the diagonal), all classes."""
    n = len(classes)
    fig, ax = plt.subplots(figsize=(11.5, 10.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [_short(c) for c in classes]
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    for i in range(n):
        for j in range(n):
            v = cm_norm[i, j]
            if v >= 0.005 and i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.5, color="#b00")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="share of the true class")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return _save(fig, out, dpi=150)


def plot_coverage(
    curves: dict[str, pd.DataFrame], threshold: float, target: float, out: str | Path
) -> Path:
    """Accuracy of the answered images vs share answered, for each split, with the threshold."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    for name, curve in curves.items():
        axes[0].plot(curve["coverage"], curve["accuracy_answered"], label=name)
        axes[1].plot(curve["threshold"], curve["coverage"], label=f"{name}: answered")
        axes[1].plot(
            curve["threshold"], curve["accuracy_answered"], ls="--", label=f"{name}: accuracy"
        )
        at = curve.iloc[(curve["threshold"] - threshold).abs().argmin()]
        axes[0].scatter([at["coverage"]], [at["accuracy_answered"]], s=25, zorder=3)
    axes[0].axhline(target, color="k", ls=":", lw=1)
    axes[0].set_xlabel("share of images answered (coverage)")
    axes[0].set_ylabel("accuracy of the answered images")
    axes[0].set_xlim(0.85, 1.0)
    axes[0].set_ylim(0.97, 1.0005)
    axes[1].axvline(threshold, color="k", ls=":", lw=1)
    axes[1].set_xlabel("confidence threshold")
    axes[1].set_xlim(0.5, 1.0)
    axes[1].set_ylim(0.8, 1.0005)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(
        f'"Not sure" below {threshold:.4f} (chosen on val, target {target:.1%})', fontsize=9
    )
    fig.tight_layout()
    return _save(fig, out, dpi=130)


def plot_cam_grid(items: list[dict], out: str | Path, title: str = "", cols: int = 3) -> Path:
    """Pairs of panels per item: the photo and its Grad-CAM overlay, with a caption.

    Each item: {"image": uint8 RGB, "overlay": uint8 RGB, "caption": str, "bad": bool}.
    """
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, 2 * cols, figsize=(2 * cols * 1.9, rows * 2.35), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for k, item in enumerate(items):
        r, c = divmod(k, cols)
        a, b = axes[r, 2 * c], axes[r, 2 * c + 1]
        a.imshow(item["image"])
        b.imshow(item["overlay"])
        colour = "#b00" if item.get("bad") else "#060"
        a.set_title(item["caption"], fontsize=6, loc="left", color=colour)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return _save(fig, out, dpi=130)


def plot_cam_on_leaf(shares: pd.DataFrame, out: str | Path) -> Path:
    """Histogram of the Grad-CAM mass on the leaf, against the leaf's share of the image."""
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bins = np.linspace(0, 1, 41)
    ax.hist(shares["leaf_area"], bins=bins, alpha=0.55, label="leaf share of the image")
    ax.hist(shares["cam_on_leaf"], bins=bins, alpha=0.65, label="Grad-CAM mass on the leaf")
    ax.axvline(shares["cam_on_leaf"].median(), color="C1", ls="--", lw=1)
    ax.axvline(shares["leaf_area"].median(), color="C0", ls="--", lw=1)
    ax.set_xlabel("share")
    ax.set_ylabel("test images")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, out, dpi=130)
