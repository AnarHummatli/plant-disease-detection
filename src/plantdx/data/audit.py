"""Near-duplicate audit across splits (plan: measure residual leakage, D-004).

Leaf ids are known for only 75.7% of the images. For the rest ("fallback") two photos of
the same physical leaf may sit in different splits without any id revealing it. To measure
this we compute a 256-bit difference hash (dHash) of every image and, per class, find for
each val/test image its nearest neighbour among the images of earlier splits.

A small Hamming distance means "almost the same picture". The same statistic computed for
images with verified leaf ids ("map") is the control: leaf-grouped splits cannot contain
the same leaf on both sides, so that rate is the baseline of look-alikes that are *not*
the same leaf. Compare the two before drawing conclusions.

Limitation: a perceptual hash finds near-identical copies (re-saved, slightly resized or
re-compressed images). It does not recognise a *different photo of the same leaf*, so a 0%
result means "no duplicate images across splits", not "no shared leaves". See D-004.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

HASH_SIDE = 16  # 16 x 16 comparisons -> 256 bits
HASH_BITS = HASH_SIDE * HASH_SIDE
HASH_WORDS = HASH_BITS // 64
THRESHOLDS = (0, 5, 10, 20)

_POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(path: str | Path) -> np.ndarray:
    """256-bit difference hash as an array of 4 uint64."""
    with Image.open(path) as im:
        im.draft("L", (HASH_SIDE * 8, HASH_SIDE * 8))  # fast JPEG downscale; no-op otherwise
        small = im.convert("L").resize((HASH_SIDE + 1, HASH_SIDE), Image.Resampling.LANCZOS)
    a = np.asarray(small, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).astype(np.uint8).ravel()
    return np.packbits(bits).view(np.uint64)


def hash_paths(paths: list[Path], workers: int = 4, desc: str = "hashing") -> np.ndarray:
    """Hash many files in parallel. Returns an (N, 4) uint64 array in input order."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(tqdm(pool.map(dhash, paths), total=len(paths), desc=desc, unit="img"))
    return np.stack(out) if out else np.zeros((0, HASH_WORDS), dtype=np.uint64)


def _popcount_sum(x: np.ndarray) -> np.ndarray:
    """Total set bits over the last axis of a uint64 array."""
    if hasattr(np, "bitwise_count"):  # NumPy >= 2.0
        return np.bitwise_count(x).sum(axis=-1, dtype=np.int16)
    as_bytes = np.ascontiguousarray(x).view(np.uint8)
    return _POPCOUNT_TABLE[as_bytes].sum(axis=-1, dtype=np.int16)


def hamming_nearest(
    queries: np.ndarray, pool: np.ndarray, chunk: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """For every query hash: smallest Hamming distance to `pool` and the pool index of it."""
    if len(pool) == 0:
        return (
            np.full(len(queries), HASH_BITS + 1, dtype=np.int16),
            np.full(len(queries), -1, dtype=np.int64),
        )
    best_d = np.empty(len(queries), dtype=np.int16)
    best_i = np.empty(len(queries), dtype=np.int64)
    for s in range(0, len(queries), chunk):
        q = queries[s : s + chunk]
        dist = _popcount_sum(q[:, None, :] ^ pool[None, :, :])
        best_i[s : s + chunk] = dist.argmin(axis=1)
        best_d[s : s + chunk] = dist.min(axis=1)
    return best_d, best_i


def nearest_cross_split(
    table: pd.DataFrame,
    hashes: np.ndarray,
    query_split: str,
    pool_splits: tuple[str, ...],
) -> pd.DataFrame:
    """Nearest same-class neighbour in `pool_splits` for every image of `query_split`.

    `table` needs columns path, label_index, split, leaf_source and its rows must be aligned
    with `hashes`. Only same-class matches are searched: a look-alike of another class is
    not leakage.
    """
    rows = []
    for label, q_group in table[table["split"] == query_split].groupby("label_index"):
        p_group = table[(table["split"].isin(pool_splits)) & (table["label_index"] == label)]
        d, idx = hamming_nearest(hashes[q_group.index.to_numpy()], hashes[p_group.index.to_numpy()])
        match_paths = p_group["path"].to_numpy()[idx] if len(p_group) else [None] * len(q_group)
        rows.append(
            pd.DataFrame(
                {
                    "query_split": query_split,
                    "query_path": q_group["path"].to_numpy(),
                    "leaf_source": q_group["leaf_source"].to_numpy(),
                    "label_index": label,
                    "distance": d,
                    "match_path": match_paths,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def summarize_matches(matches: pd.DataFrame, thresholds=THRESHOLDS) -> pd.DataFrame:
    """Share of query images with a same-class neighbour at distance <= t, by source."""
    out = []
    for (split, source), g in matches.groupby(["query_split", "leaf_source"]):
        row = {"query_split": split, "leaf_source": source, "images": len(g)}
        for t in thresholds:
            row[f"<= {t}"] = round(float((g["distance"] <= t).mean()) * 100, 2)
        out.append(row)
    return pd.DataFrame(out)


def positive_control(
    dropped: pd.DataFrame,
    dropped_hashes: np.ndarray,
    table: pd.DataFrame,
    hashes: np.ndarray,
    against: str = "test",
) -> pd.DataFrame:
    """Nearest same-class `against` image for every train image removed by the split builder.

    These removed images share a (fallback) leaf id with a test image, so if any image pairs
    in the dataset are near-identical, these are the most likely ones. Running the same
    search on them shows what the audit reports when duplicates *do* exist, which is
    what gives meaning to a 0% result on the real splits. `dropped` rows must be aligned
    with `dropped_hashes` and `table` rows with `hashes`.
    """
    rows = []
    dropped = dropped.reset_index(drop=True)
    for label, group in dropped.groupby("label_index"):
        pool = table[(table["split"] == against) & (table["label_index"] == label)]
        d, idx = hamming_nearest(
            dropped_hashes[group.index.to_numpy()], hashes[pool.index.to_numpy()]
        )
        match_paths = pool["path"].to_numpy()[idx] if len(pool) else [None] * len(group)
        rows.append(
            pd.DataFrame(
                {
                    "query_split": "dropped_train",
                    "query_path": group["path"].to_numpy(),
                    "leaf_source": group["leaf_source"].to_numpy(),
                    "label_index": label,
                    "distance": d,
                    "match_path": match_paths,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def distance_quantiles(matches: pd.DataFrame, by: str = "leaf_source") -> pd.DataFrame:
    """Min and low quantiles of the nearest-neighbour distance per group (rounded)."""
    q = matches.groupby(by)["distance"]
    out = pd.DataFrame(
        {
            "images": q.size(),
            "min": q.min(),
            "p01": q.quantile(0.01),
            "p05": q.quantile(0.05),
            "median": q.median(),
            "max": q.max(),
        }
    )
    return out.round(1).reset_index()
