from pathlib import Path

import numpy as np

import plantdx
from plantdx.config import deep_merge, load_config, set_seed
from plantdx.paths import ENV_ARTIFACTS, artifacts_dir, repo_root

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_package_imports():
    assert plantdx.__version__


def test_base_config_loads():
    cfg = load_config(CONFIGS / "base.yaml")
    assert cfg["seed"] == 42
    assert cfg["data"]["num_classes"] == 38
    assert cfg["data"]["image_size"] == 224


def test_extends_merges_nested_keys(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 42\ntrain: {batch_size: 64, epochs: 10}\n")
    (tmp_path / "exp.yaml").write_text("extends: base.yaml\ntrain: {epochs: 3}\n")
    cfg = load_config(tmp_path / "exp.yaml")
    assert cfg == {"seed": 42, "train": {"batch_size": 64, "epochs": 3}}


def test_deep_merge_does_not_mutate_inputs():
    base = {"a": {"b": 1}}
    deep_merge(base, {"a": {"b": 2}})
    assert base == {"a": {"b": 1}}


def test_set_seed_is_reproducible():
    set_seed(7)
    a = np.random.rand(3)
    set_seed(7)
    b = np.random.rand(3)
    assert np.allclose(a, b)


def test_artifacts_dir_respects_env(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_ARTIFACTS, str(tmp_path / "drive"))
    assert artifacts_dir() == tmp_path / "drive"
    assert (tmp_path / "drive").is_dir()


def test_repo_root_has_pyproject():
    assert (repo_root() / "pyproject.toml").exists()


def test_cache_dir_env_and_fallback(tmp_path, monkeypatch):
    from plantdx.paths import ENV_CACHE, cache_dir

    monkeypatch.setenv(ENV_CACHE, str(tmp_path / "fast"))
    assert cache_dir() == tmp_path / "fast" and (tmp_path / "fast").is_dir()
    monkeypatch.delenv(ENV_CACHE)
    monkeypatch.setenv(ENV_ARTIFACTS, str(tmp_path / "art"))
    expected = Path("/content/cache") if Path("/content").is_dir() else tmp_path / "art" / "cache"
    assert cache_dir(create=False) == expected


def test_package_version_matches_pyproject():
    import re

    text = (CONFIGS.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "([^"]+)"', text, re.M).group(1) == plantdx.__version__
