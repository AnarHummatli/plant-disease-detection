"""Model builders and the trainer (plan steps 3.3, 3.3a; test 3.4). Need TensorFlow."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

tf = pytest.importorskip("tensorflow")

import keras  # noqa: E402

from plantdx.config import load_config  # noqa: E402
from plantdx.data.source import CLASSES  # noqa: E402
from plantdx.models import build_model  # noqa: E402
from plantdx.models.cnn import build_cnn  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_cnn_config_builds_expected_model():
    cfg = load_config(CONFIGS / "cnn.yaml")
    assert cfg["model"]["name"] == "cnn" and cfg["seed"] == 42  # inherits base.yaml
    model = build_model(cfg)
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 38)
    assert model.count_params() == 1_184_902


def test_cnn_output_is_a_distribution_and_takes_raw_pixels():
    model = build_cnn(image_size=32, num_classes=5, filters=(4, 8), convs_per_block=1)
    x = np.random.default_rng(0).uniform(0, 255, (3, 32, 32, 3)).astype(np.float32)
    p = model.predict(x, verbose=0)
    assert p.shape == (3, 5) and np.allclose(p.sum(axis=1), 1, atol=1e-5)
    assert any(isinstance(layer, keras.layers.Rescaling) for layer in model.layers)


def test_augmentation_runs_only_in_training():
    from plantdx.data.pipeline import AugmentConfig, build_augmentation

    aug = build_augmentation(AugmentConfig(), seed=1)
    model = build_cnn(image_size=32, num_classes=3, filters=(4,), augmentation=aug)
    x = np.random.default_rng(0).uniform(0, 255, (2, 32, 32, 3)).astype(np.float32)
    a = model(x, training=False).numpy()
    b = model(x, training=False).numpy()
    assert np.allclose(a, b)  # inference is deterministic: augmentation is off


def test_unknown_model_name():
    with pytest.raises(ValueError):
        build_model({"model": {"name": "resnet"}})


def make_fake_project(tmp_path, n_classes=4):
    """Tiny images for the first `n_classes` classes and train/val/test CSVs."""
    rng = np.random.default_rng(0)
    splits = tmp_path / "splits"
    splits.mkdir()
    for split, n in (("train", 6), ("val", 3), ("test", 2)):
        rows = []
        for i, label in enumerate(CLASSES[:n_classes]):
            for j in range(n):
                rel = f"{label}/{split}{j}.jpg"
                p = tmp_path / "data" / "raw" / "color" / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                img = np.full((24, 24, 3), 40 * i, np.uint8) + rng.integers(0, 30, (24, 24, 3))
                Image.fromarray(img.astype(np.uint8)).save(p)
                leaf = f"{label}:{split}{j}"
                rows.append(
                    {
                        "path": rel,
                        "label": label,
                        "label_index": i,
                        "leaf_id": leaf,
                        "group_id": leaf,
                        "leaf_source": "map",
                    }
                )
        pd.DataFrame(rows).to_csv(splits / f"{split}.csv", index=False)
    cfg = tmp_path / "exp.yaml"
    cfg.write_text(
        f"extends: {CONFIGS / 'cnn.yaml'}\n"
        "run_id: smoke\n"
        "data: {image_size: 16, cache_train: null, cache_eval: null}\n"
        "model: {filters: [4], convs_per_block: 1}\n"
        "train: {batch_size: 8, epochs: 1}\n",
        encoding="utf-8",
    )
    return cfg, tmp_path / "data", splits


def test_trainer_one_epoch_writes_run_and_results(tmp_path, monkeypatch):
    from plantdx.train import train

    monkeypatch.setenv("PLANTDX_ARTIFACTS", str(tmp_path / "artifacts"))
    cfg, data, splits = make_fake_project(tmp_path)
    report = train(cfg, data, splits_dir=splits, verbose=0)

    run = tmp_path / "artifacts" / "runs" / "smoke"
    for name in ("config.yaml", "env.json", "best.keras", "history.csv", "metrics_val.json"):
        assert (run / name).is_file(), name
    assert report["epochs_trained"] == 1 and report["metrics"]["n"] == 12
    saved = json.loads((run / "metrics_val.json").read_text())
    assert len(saved["per_class"]) == 38
    scores = np.load(run / "val_scores.npz")
    assert scores["probs"].shape == (12, 38)

    results = pd.read_csv(tmp_path / "artifacts" / "reports" / "results.csv")
    assert results["run_id"].tolist() == ["smoke"] and results["split"].tolist() == ["val"]
    history = pd.read_csv(run / "history.csv")
    assert {"loss", "val_loss", "val_macro_f1", "val_top3"} <= set(history.columns)
