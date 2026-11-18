"""Recover leaf groups for images without a verified leaf id (D-010).

Problem. PlantVillage photographed most leaves about 4 times (median of the verified leaves).
For 24% of the images ("fallback") the leaf id comes only from the file name, and 97% of those
ids contain a single image, so the other photos of the same leaf are not grouped and can sit in
train and test at the same time. The near-identical search (D-006) does not help: photos of one
leaf are usually taken in different poses, not as near-copies.

Method.
1. Embed every image with an ImageNet-pretrained CNN (MobileNetV2, global average pooling),
   optionally averaged over the 8 rotations/flips (D4) so that the pose matters less.
2. Inside each class, take the k nearest neighbours of every image (cosine similarity).
3. **Calibrate on the verified leaves**: among neighbour pairs of two verified images, whether
   they are the same leaf is known, so precision and recall of "similarity >= t" are measured.
4. Pairs above the chosen cut-off that involve a fallback image are joined into groups by the
   split builder, exactly like the near-identical pairs of D-006.

The calibration is the point: the cut-off is chosen where the method is shown to be right on
known leaves, not assumed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from plantdx.data.duplicates import D4, apply_transform


def build_embedder(image_size: int = 224):
    """MobileNetV2 (ImageNet) feature extractor: raw 0-255 RGB in, 1280-d vector out."""
    import keras

    inputs = keras.Input((image_size, image_size, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    base = keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
        pooling="avg",
    )
    return keras.Model(inputs, base(x, training=False), name="embedder")


def _l2(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)


def embed(dataset, model, tta: str = "none", total_batches: int | None = None) -> np.ndarray:
    """L2-normalised embeddings (float32) for every image of an un-shuffled tf.data dataset.

    tta="d4" averages the normalised embeddings of the 8 rotations/flips of each image.
    """
    if tta not in ("none", "d4"):
        raise ValueError("tta must be 'none' or 'd4'")
    transforms = D4 if tta == "d4" else ("identity",)
    out = []
    for x, _ in tqdm(dataset, total=total_batches, desc=f"embed ({tta})", unit="batch"):
        x = x.numpy()
        acc = None
        for t in transforms:
            xt = np.ascontiguousarray(np.moveaxis(apply_transform(np.moveaxis(x, -1, 1), t), 1, -1))
            e = _l2(np.asarray(model(xt, training=False), dtype=np.float32))
            acc = e if acc is None else acc + e
        out.append(_l2(acc))
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


def knn_pairs(
    table: pd.DataFrame, emb: np.ndarray, k: int = 10, by: str = "label_index", chunk: int = 2048
) -> pd.DataFrame:
    """Undirected pairs (a, b) where b is among a's k most similar images of the same class.

    `table` rows must be aligned with `emb`. Columns: path_a, path_b, label_index, sim, mutual
    (True if each is in the other's top k).
    """
    paths = table["path"].to_numpy()
    rows = []
    for label, idx in table.groupby(by, sort=True).indices.items():
        if len(idx) < 2:
            continue
        e = emb[idx]
        kk = min(k, len(idx) - 1)
        nbr = np.empty((len(idx), kk), dtype=np.int64)
        sim = np.empty((len(idx), kk), dtype=np.float32)
        for s in range(0, len(idx), chunk):
            block = e[s : s + chunk] @ e.T
            block[np.arange(len(block)), np.arange(s, s + len(block))] = -np.inf  # not itself
            part = np.argpartition(-block, kk - 1, axis=1)[:, :kk]
            nbr[s : s + chunk] = part
            sim[s : s + chunk] = np.take_along_axis(block, part, axis=1)
        a = np.repeat(np.arange(len(idx)), kk)
        b = nbr.ravel()
        directed = set(zip(a.tolist(), b.tolist(), strict=True))
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        frame = pd.DataFrame({"i": lo, "j": hi, "sim": sim.ravel()})
        frame = frame.groupby(["i", "j"], as_index=False)["sim"].max()
        frame["mutual"] = [
            (i, j) in directed and (j, i) in directed for i, j in zip(frame.i, frame.j, strict=True)
        ]
        rows.append(
            pd.DataFrame(
                {
                    "path_a": paths[idx[frame.i]],
                    "path_b": paths[idx[frame.j]],
                    "label_index": label,
                    "sim": frame["sim"].round(4).to_numpy(),
                    "mutual": frame["mutual"].to_numpy(),
                }
            )
        )
    cols = ["path_a", "path_b", "label_index", "sim", "mutual"]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=cols)


def annotate(pairs: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Add leaf ids, sources and splits of both images; `known` = both verified, `same_leaf`."""
    info = table.set_index("path")[["leaf_id", "leaf_source", "split"]]
    out = pairs.join(info.add_suffix("_a"), on="path_a").join(info.add_suffix("_b"), on="path_b")
    out["known"] = (out["leaf_source_a"] == "map") & (out["leaf_source_b"] == "map")
    out["same_leaf"] = out["leaf_id_a"] == out["leaf_id_b"]
    out["involves_fallback"] = (out["leaf_source_a"] == "fallback") | (
        out["leaf_source_b"] == "fallback"
    )
    out["cross_split"] = out["split_a"] != out["split_b"]
    return out


def calibration_curve(
    pairs: pd.DataFrame,
    table: pd.DataFrame,
    thresholds=None,
    mutual_only: bool = False,
) -> pd.DataFrame:
    """Precision and recall of "sim >= t" on pairs of two verified (map) images.

    precision = same-leaf pairs / all known pairs above t
    recall    = same-leaf pairs above t / all same-leaf pairs among verified images
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.50, 0.981, 0.02), 2)
    known = pairs[pairs["known"]]
    if mutual_only:
        known = known[known["mutual"]]
    sizes = table[table["leaf_source"] == "map"].groupby("leaf_id").size()
    total_same = int((sizes * (sizes - 1) // 2).sum())
    rows = []
    for t in thresholds:
        sel = known[known["sim"] >= t]
        tp = int(sel["same_leaf"].sum())
        rows.append(
            {
                "threshold": float(t),
                "pairs": len(sel),
                "same_leaf": tp,
                "precision": round(tp / len(sel), 4) if len(sel) else float("nan"),
                "recall": round(tp / total_same, 4) if total_same else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def pick_threshold(curve: pd.DataFrame, min_precision: float = 0.95) -> float | None:
    """Lowest threshold whose precision (and every higher one's) is >= min_precision."""
    ok = curve.dropna(subset=["precision"]).sort_values("threshold")
    good = (ok["precision"] >= min_precision).to_numpy()
    # precision must stay above the bar from t upwards, not just touch it once
    stays = np.minimum.accumulate(good[::-1])[::-1]
    for t, s in zip(ok["threshold"], stays, strict=True):
        if s:
            return float(t)
    return None


def group_size_stats(table: pd.DataFrame, groups: pd.Series, mask: pd.Series) -> dict:
    """Size distribution of groups among the rows selected by `mask` (e.g. fallback images)."""
    sizes = groups[mask].value_counts()
    return {
        "groups": int(len(sizes)),
        "images": int(sizes.sum()),
        "mean": round(float(sizes.mean()), 2) if len(sizes) else 0.0,
        "median": float(sizes.median()) if len(sizes) else 0.0,
        "max": int(sizes.max()) if len(sizes) else 0,
        "single_image_share": round(float((sizes == 1).mean()), 4) if len(sizes) else 0.0,
    }


def save_embeddings(path: str | Path, emb: np.ndarray, paths: list[str]) -> None:
    np.savez_compressed(path, emb=emb.astype(np.float16), paths=np.array(paths))


def load_embeddings(path: str | Path, paths: list[str]) -> np.ndarray | None:
    """Cached embeddings if they belong to exactly these paths (in this order), else None."""
    path = Path(path)
    if not path.exists():
        return None
    z = np.load(path)
    if z["paths"].tolist() != list(paths):
        return None
    return z["emb"].astype(np.float32)
