"""Where things live on disk.

Artifacts (trained models, logs, cached features) must survive a Colab session
reset, so on Colab they go to Google Drive. The location is controlled by one
environment variable, PLANTDX_ARTIFACTS, which the bootstrap notebook sets.
Locally it falls back to ./artifacts inside the repo (git-ignored).
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_ARTIFACTS = "PLANTDX_ARTIFACTS"


def repo_root() -> Path:
    """Repository root: the folder that contains pyproject.toml."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Installed as a regular (non-editable) package: fall back to the cwd.
    return Path.cwd()


def artifacts_dir(create: bool = True) -> Path:
    """Root folder for models, logs and caches."""
    env = os.environ.get(ENV_ARTIFACTS)
    path = Path(env).expanduser() if env else repo_root() / "artifacts"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(run_id: str, create: bool = True) -> Path:
    """Folder for one experiment run, e.g. artifacts/runs/E2_mobilenetv2_head."""
    path = artifacts_dir(create=create) / "runs" / run_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


ENV_CACHE = "PLANTDX_CACHE"


def cache_dir(create: bool = True) -> Path:
    """Fast local disk for tf.data cache files (several GB; never Google Drive).

    PLANTDX_CACHE if set; on Colab /content/cache (the VM's local disk); otherwise
    <artifacts>/cache.
    """
    env = os.environ.get(ENV_CACHE)
    if env:
        path = Path(env).expanduser()
    elif Path("/content").is_dir():
        path = Path("/content/cache")
    else:
        path = artifacts_dir(create=create) / "cache"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
