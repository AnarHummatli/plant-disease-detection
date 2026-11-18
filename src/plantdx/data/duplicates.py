"""Find near-identical images anywhere in the dataset (replaces the dHash threshold audit).

"Near-identical" covers copies of one photo (re-saved, blurred, resized, flipped, rotated) and
re-shots of the same leaf taken moments apart. On PlantVillage the pairs that cross the splits
turned out to be re-shots (D-006).

Why a new method. The first audit (dHash, Hamming distance <= 20) reported 0% cross-split
duplicates, but looking at the closest pairs by eye showed identical photos 30-34 bits apart:
on flat grey backgrounds a slight blur or re-compression flips many hash bits. See D-004.

Method, per class, over ALL images (all splits together, so the result does not depend on
how the splits were drawn):

1. Stage 1 (fast filter). Every image becomes a 32x32 greyscale thumbnail, normalised to zero
   mean and unit length, so a dot product is the Pearson correlation. All pairs are compared
   with one matrix product per transform of the dihedral group D4 (identity, flips, 90-degree
   rotations, transposes); pairs whose best correlation is >= `min_corr` become candidates.
2. Stage 2 (verification). Candidates are re-compared at 96x96 with the transform that matched.
   Copies of one photo stay highly correlated at the higher resolution; different leaves that
   merely look alike at 32x32 do not.

A pair (a, b, t) means: image a looks like transform t applied to image b. The final cut-off on
the stage-2 correlation is chosen after looking at example pairs (notebook 02) and recorded in
splits/duplicates_meta.json.

Limitation: correlation does not cleanly separate re-shots of one leaf from different leaves
that look alike (both reach about 0.92-0.95 on PlantVillage), and photos of the same leaf from a
clearly different angle are not found. The cut-off is therefore chosen conservatively (D-006).
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

D4 = (
    "identity",
    "hflip",
    "vflip",
    "rot180",
    "rot90",
    "rot270",
    "transpose",
    "antitranspose",
)


def apply_transform(x: np.ndarray, name: str) -> np.ndarray:
    """Apply a D4 transform to the last two axes of `x` (square images)."""
    if name == "identity":
        return x
    if name == "hflip":
        return x[..., :, ::-1]
    if name == "vflip":
        return x[..., ::-1, :]
    if name == "rot180":
        return x[..., ::-1, ::-1]
    if name == "rot90":
        return np.rot90(x, 1, axes=(-2, -1))
    if name == "rot270":
        return np.rot90(x, 3, axes=(-2, -1))
    if name == "transpose":
        return np.swapaxes(x, -2, -1)
    if name == "antitranspose":
        return np.swapaxes(x[..., ::-1, ::-1], -2, -1)
    raise ValueError(f"unknown transform {name!r}; expected one of {D4}")


def load_gray(path: str | Path, size: int) -> np.ndarray:
    """Greyscale `size` x `size` uint8 array (aspect ratio is not kept; images are square)."""
    with Image.open(path) as im:
        im.draft("L", (size * 2, size * 2))  # fast JPEG downscale; no-op for other formats
        small = im.convert("L").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(small, dtype=np.uint8)


def thumbnails(
    paths: Sequence[str | Path], size: int = 32, workers: int = 4, desc: str = "thumbnails"
) -> np.ndarray:
    """(N, size, size) uint8 thumbnails, loaded in parallel, in input order."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(
            tqdm(
                pool.map(lambda p: load_gray(p, size), paths),
                total=len(paths),
                desc=desc,
                unit="img",
            )
        )
    return np.stack(out) if out else np.zeros((0, size, size), dtype=np.uint8)


def normalize(x: np.ndarray) -> np.ndarray:
    """Flatten the last two axes; zero mean, unit L2 norm (flat images become all zeros)."""
    flat = x.reshape(*x.shape[:-2], -1).astype(np.float32)
    flat -= flat.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(flat, axis=-1, keepdims=True)
    return np.divide(flat, norm, out=np.zeros_like(flat), where=norm > 0)


def candidate_pairs(
    table: pd.DataFrame,
    thumbs: np.ndarray,
    min_corr: float = 0.90,
    transforms: Sequence[str] = D4,
    chunk: int = 1024,
    max_pairs: int = 2_000_000,
    by: str = "label_index",
) -> pd.DataFrame:
    """Stage 1: all same-class pairs whose best 32x32 correlation over `transforms` >= min_corr.

    `table` (columns path, label_index) must be aligned with `thumbs` by position.
    Returns columns path_a, path_b, label_index, corr32, transform.
    """
    paths = table["path"].to_numpy()
    groups = table.groupby(by, sort=True).indices
    parts, total = [], 0
    for label, idx in tqdm(groups.items(), desc="stage 1", unit="class"):
        if len(idx) < 2:
            continue
        base = normalize(thumbs[idx])
        variants = [normalize(apply_transform(thumbs[idx], t)) for t in transforms]
        for s in range(0, len(idx), chunk):
            rows = base[s : s + chunk]
            best = np.full((len(rows), len(idx)), -np.inf, dtype=np.float32)
            best_t = np.zeros(best.shape, dtype=np.int8)
            for k, var in enumerate(variants):
                block = rows @ var.T
                better = block > best
                best[better] = block[better]
                best_t[better] = k
            i_local, j = np.nonzero(best >= min_corr)
            i = i_local + s
            keep = j > i  # each unordered pair once, never an image with itself
            i, j, i_local = i[keep], j[keep], i_local[keep]
            if len(i) == 0:
                continue
            total += len(i)
            if total > max_pairs:
                raise RuntimeError(
                    f"more than {max_pairs:,} candidate pairs at min_corr={min_corr}; "
                    "raise min_corr"
                )
            parts.append(
                pd.DataFrame(
                    {
                        "path_a": paths[idx[i]],
                        "path_b": paths[idx[j]],
                        "label_index": label,
                        "corr32": best[i_local, j].round(4),
                        "transform": np.asarray(transforms)[best_t[i_local, j]],
                    }
                )
            )
    cols = ["path_a", "path_b", "label_index", "corr32", "transform"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def verify_pairs(
    pairs: pd.DataFrame,
    load,
    size: int = 96,
    workers: int = 4,
) -> pd.DataFrame:
    """Stage 2: add `corr_hi`, the correlation at `size` x `size` with the matched transform.

    `load(path, size)` returns a greyscale uint8 array; in the notebook it reads the image file.
    """
    pairs = pairs.copy()
    if pairs.empty:
        pairs["corr_hi"] = pd.Series(dtype=float)
        return pairs
    unique = pd.unique(pd.concat([pairs["path_a"], pairs["path_b"]]))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        arrays = list(
            tqdm(
                pool.map(lambda p: load(p, size), unique),
                total=len(unique),
                desc="stage 2",
                unit="img",
            )
        )
    lookup = dict(zip(unique, arrays, strict=True))
    corr = []
    for a, b, t in zip(pairs["path_a"], pairs["path_b"], pairs["transform"], strict=True):
        va = normalize(lookup[a])
        vb = normalize(np.ascontiguousarray(apply_transform(lookup[b], t)))
        corr.append(float(va @ vb))
    pairs["corr_hi"] = np.round(corr, 4)
    return pairs


def annotate(pairs: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Add split / leaf columns for both images and flags `cross_split`, `same_leaf`."""
    info = table.set_index("path")[["split", "leaf_id", "leaf_source"]]
    out = pairs.join(info.add_suffix("_a"), on="path_a").join(info.add_suffix("_b"), on="path_b")
    out["cross_split"] = out["split_a"] != out["split_b"]
    out["same_leaf"] = out["leaf_id_a"] == out["leaf_id_b"]
    return out


def summarize_duplicates(dups: pd.DataFrame) -> pd.DataFrame:
    """Counts of confirmed duplicate pairs by split pair, for the report."""
    if dups.empty:
        return pd.DataFrame(columns=["splits", "pairs", "different_leaf_id", "fallback_involved"])
    key = dups.apply(lambda r: "&".join(sorted((r["split_a"], r["split_b"]))), axis=1)
    fb = (dups["leaf_source_a"] == "fallback") | (dups["leaf_source_b"] == "fallback")
    out = (
        pd.DataFrame({"splits": key, "diff": ~dups["same_leaf"], "fb": fb})
        .groupby("splits")
        .agg(
            pairs=("diff", "size"),
            different_leaf_id=("diff", "sum"),
            fallback_involved=("fb", "sum"),
        )
        .reset_index()
    )
    return out
