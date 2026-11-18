"""Download data.zip and extract one variant's images (plan step 2.1).

data.zip is about 2.2 GB and contains all three variants (color, grayscale, segmented).
We extract only `raw/<variant>/...` to a fast local folder (on Colab: /content/data, not
Drive, because reading thousands of small files from Drive is slow).
"""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from pathlib import Path

from tqdm.auto import tqdm

from plantdx.data.source import DATA_ZIP_FILE, HF_REPO_ID, HF_REPO_TYPE, zip_member


def download_archive(cache_dir: str | Path | None = None) -> Path:
    """Download data.zip from the Hub (cached; a second call is instant)."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            filename=DATA_ZIP_FILE,
            cache_dir=cache_dir,
        )
    )


def extract_variant(zip_path: str | Path, dest: str | Path, variant: str = "color") -> int:
    """Extract `raw/<variant>/*` into `dest`. Skips files that are already there.

    Returns the number of files newly written.
    """
    prefix = f"raw/{variant}/"
    dest = Path(dest)
    written = 0
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if m.filename.startswith(prefix) and not m.is_dir()]
        if not members:
            raise FileNotFoundError(f"no members under {prefix!r} in {zip_path}")
        for m in tqdm(members, desc=f"extract {variant}", unit="img"):
            target = dest / m.filename
            if target.exists() and target.stat().st_size == m.file_size:
                continue
            zf.extract(m, dest)  # ZipFile sanitises '..' and absolute paths
            written += 1
    return written


def image_path(root: str | Path, variant: str, rel_path: str) -> Path:
    """Local path of an image given its splits/*.csv `path` value."""
    return Path(root) / zip_member(variant, rel_path)


def missing_files(root: str | Path, variant: str, rel_paths: Iterable[str]) -> list[str]:
    """Entries of `rel_paths` that are not present on disk (should be empty)."""
    return [p for p in rel_paths if not image_path(root, variant, p).is_file()]
