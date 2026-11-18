"""Environment report: run `python -m plantdx.env` at the start of every session.

The output goes into the report's reproducibility section and helps when
Colab silently changes its runtime (Python / TensorFlow versions).
"""

from __future__ import annotations

import platform
import sys
from importlib import metadata

from plantdx import __version__
from plantdx.paths import artifacts_dir

PACKAGES = ("numpy", "pandas", "scikit-learn", "tensorflow", "keras", "gradio", "huggingface_hub")


def _version(pkg: str) -> str:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return "not installed"


def gpu_names() -> list[str]:
    try:
        import tensorflow as tf
    except ImportError:
        return []
    return [d.name for d in tf.config.list_physical_devices("GPU")]


def report() -> dict[str, object]:
    info: dict[str, object] = {
        "plantdx": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    info.update({pkg: _version(pkg) for pkg in PACKAGES})
    info["gpus"] = gpu_names() or "none"
    info["artifacts_dir"] = str(artifacts_dir(create=False))
    return info


def main() -> None:
    for key, value in report().items():
        print(f"{key:>16}: {value}")


if __name__ == "__main__":
    main()
