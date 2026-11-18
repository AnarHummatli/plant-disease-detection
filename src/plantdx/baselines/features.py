"""Hand-crafted image features for the classical baseline (plan step 3.1).

Each image is resized to S x S (default 128) and described by three classic descriptors:

* **colour**: HSV histogram, 32 bins per channel, each channel normalised to sum 1 (96 values).
  Disease symptoms are often colour changes (yellowing, brown or black lesions, rust).
* **shape / texture edges**: HOG on the greyscale image, 9 orientations, 16 x 16 pixel cells,
  2 x 2 cell blocks, L2-Hys (1,764 values at 128 px).
* **micro-texture**: uniform LBP (P = 8, R = 1) histogram, 10 bins, normalised (10 values).

1,870 values per image at 128 px. Extraction runs once per split and is cached as .npz under the
artifacts folder (Drive on Colab); the cache name contains the image size, the feature version and
a hash of the file list, so a changed split or feature definition never reuses a stale cache.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern

from plantdx.data.images import image_path

# Bump when the feature definition changes: old caches are then ignored.
FEATURE_VERSION = 1

HSV_BINS = 32
HOG_ORIENTATIONS = 9
HOG_CELL = 16
HOG_BLOCK = 2
LBP_P, LBP_R = 8, 1
LBP_BINS = LBP_P + 2  # "uniform" LBP codes are 0..P+1


def feature_dim(size: int = 128) -> int:
    cells = size // HOG_CELL
    blocks = max(cells - HOG_BLOCK + 1, 0)
    return 3 * HSV_BINS + blocks * blocks * HOG_BLOCK * HOG_BLOCK * HOG_ORIENTATIONS + LBP_BINS


def feature_slices(size: int = 128) -> dict[str, slice]:
    """Where each descriptor sits in the feature vector (for ablations and plots)."""
    hsv = 3 * HSV_BINS
    hog_len = feature_dim(size) - hsv - LBP_BINS
    return {
        "hsv": slice(0, hsv),
        "hog": slice(hsv, hsv + hog_len),
        "lbp": slice(hsv + hog_len, hsv + hog_len + LBP_BINS),
    }


def load_rgb(path: str | Path, size: int = 128) -> np.ndarray:
    """uint8 [size, size, 3]; any mode (RGBA, L) is converted to RGB."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.size != (size, size):
            im = im.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def image_features(rgb: np.ndarray) -> np.ndarray:
    """Feature vector (float32) of one uint8 RGB image [S, S, 3]."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected uint8 [S, S, 3], got {rgb.dtype} {rgb.shape}")
    n_pixels = rgb.shape[0] * rgb.shape[1]

    hsv = rgb2hsv(rgb)
    colour = np.concatenate(
        [np.histogram(hsv[..., c], bins=HSV_BINS, range=(0.0, 1.0))[0] for c in range(3)]
    ) / float(n_pixels)

    gray = rgb2gray(rgb)  # float in [0, 1]
    edges = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=(HOG_CELL, HOG_CELL),
        cells_per_block=(HOG_BLOCK, HOG_BLOCK),
        block_norm="L2-Hys",
        feature_vector=True,
    )

    gray_u8 = np.round(gray * 255).astype(np.uint8)
    codes = local_binary_pattern(gray_u8, LBP_P, LBP_R, method="uniform")
    texture = np.bincount(codes.astype(np.int64).ravel(), minlength=LBP_BINS)[:LBP_BINS]
    texture = texture / float(n_pixels)

    return np.concatenate([colour, edges, texture]).astype(np.float32)


def _features_for_files(files: list[str], size: int) -> np.ndarray:
    return np.stack([image_features(load_rgb(f, size)) for f in files])


def features_for_files(
    files: list[str], size: int = 128, n_jobs: int = -1, chunk: int = 512, progress: bool = True
) -> np.ndarray:
    """[len(files), feature_dim(size)] float32, computed in parallel chunks (joblib)."""
    from joblib import Parallel, delayed

    if not files:
        return np.zeros((0, feature_dim(size)), np.float32)
    chunks = [files[i : i + chunk] for i in range(0, len(files), chunk)]
    if progress:
        from tqdm.auto import tqdm

        chunks_iter = tqdm(chunks, desc=f"features ({len(files):,} images)", unit="chunk")
    else:
        chunks_iter = chunks
    parts = Parallel(n_jobs=n_jobs)(delayed(_features_for_files)(c, size) for c in chunks_iter)
    return np.concatenate(parts).astype(np.float32)


def cache_name(paths: list[str], size: int, tag: str = "") -> str:
    digest = hashlib.sha1("\n".join(paths).encode()).hexdigest()[:12]
    return f"{tag + '_' if tag else ''}{size}px_v{FEATURE_VERSION}_{digest}.npz"


def split_features(
    frame: pd.DataFrame,
    root: str | Path,
    variant: str = "color",
    size: int = 128,
    cache_dir: str | Path | None = None,
    tag: str = "",
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) for one split, read from / written to `cache_dir` if given."""
    paths = frame["path"].tolist()
    y = frame["label_index"].to_numpy(dtype=np.int64)
    cache = Path(cache_dir) / cache_name(paths, size, tag) if cache_dir else None
    if cache is not None and cache.exists():
        with np.load(cache, allow_pickle=False) as npz:
            if npz["paths"].tolist() == paths:
                return npz["X"], npz["y"]
    files = [str(image_path(root, variant, p)) for p in paths]
    X = features_for_files(files, size=size, n_jobs=n_jobs)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp.npz")
        np.savez(tmp, X=X, y=y, paths=np.array(paths))
        tmp.replace(cache)  # atomic: an interrupted run never leaves a half-written cache
    return X, y
