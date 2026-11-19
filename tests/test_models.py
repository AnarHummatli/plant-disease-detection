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
    for name in (
        "config.yaml",
        "env.json",
        "best.keras",
        "inference.keras",
        "history.csv",
        "metrics_val.json",
    ):
        assert (run / name).is_file(), name
    # the reported size is the inference model, smaller than the checkpoint with optimizer state
    assert report["size_mb"] == round((run / "inference.keras").stat().st_size / 1e6, 2)
    assert (run / "inference.keras").stat().st_size < (run / "best.keras").stat().st_size
    assert report["epochs_trained"] == 1 and report["metrics"]["n"] == 12
    saved = json.loads((run / "metrics_val.json").read_text())
    assert len(saved["per_class"]) == 38
    scores = np.load(run / "val_scores.npz")
    assert scores["probs"].shape == (12, 38)

    results = pd.read_csv(tmp_path / "artifacts" / "reports" / "results.csv")
    assert results["run_id"].tolist() == ["smoke"] and results["split"].tolist() == ["val"]
    history = pd.read_csv(run / "history.csv")
    assert {"loss", "val_loss", "val_macro_f1", "val_top3"} <= set(history.columns)


def test_inference_export_matches_mixed_precision_model(tmp_path):
    """float32 copy without augmentation predicts like the mixed-precision trained model."""
    from plantdx.export import INFERENCE_FILE, export_inference

    cfg = {
        "data": {"image_size": 32, "num_classes": 5},
        "model": {"name": "cnn", "filters": [4, 8], "convs_per_block": 2},
        "augment": {},
    }
    from plantdx.data.pipeline import AugmentConfig, build_augmentation

    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        trained = build_model(cfg, build_augmentation(AugmentConfig(), seed=1))
    finally:
        keras.mixed_precision.set_global_policy("float32")
    trained.compile(optimizer="adam", loss="categorical_crossentropy")
    x = np.random.default_rng(0).uniform(0, 255, (4, 32, 32, 3)).astype(np.float32)
    trained.fit(x, np.eye(5)[[0, 1, 2, 3]], epochs=1, verbose=0)  # creates optimizer state

    path = export_inference(cfg, trained, tmp_path)
    assert path.name == INFERENCE_FILE
    clean = keras.models.load_model(path)
    assert all(layer.dtype_policy.name == "float32" for layer in clean.layers)
    assert "augmentation" not in [layer.name for layer in clean.layers]
    assert getattr(clean, "optimizer", None) is None
    np.testing.assert_allclose(
        clean.predict(x, verbose=0), trained.predict(x, verbose=0), atol=2e-3
    )


def test_copy_weights_by_name_rejects_mismatch():
    from plantdx.export import copy_weights_by_name

    a = build_cnn(image_size=16, num_classes=3, filters=(4,), convs_per_block=1)
    b = build_cnn(image_size=16, num_classes=3, filters=(8,), convs_per_block=1)
    with pytest.raises(ValueError):
        copy_weights_by_name(a, b)


# ---------------------------------------------------------------- MobileNetV2 (stage 4)
def test_mobilenetv2_head_and_fine_tune_trainable_parts():
    from plantdx.models.mobilenet import BACKBONE, build_mobilenetv2

    def n_trainable(m):
        return sum(int(np.prod(w.shape)) for w in m.trainable_weights)

    head = build_mobilenetv2(weights=None)
    assert head.output_shape == (None, 38) and head.count_params() == 2_306_662
    assert n_trainable(head) == 48_678  # GAP -> Dense(38) only

    ft = build_mobilenetv2(weights=None, fine_tune_from="block_13_expand")
    assert n_trainable(ft) == 1_712_038
    backbone = ft.get_layer(BACKBONE)
    names = [layer.name for layer in backbone.layers]
    start = names.index("block_13_expand")
    for i, layer in enumerate(backbone.layers):
        if isinstance(layer, keras.layers.BatchNormalization) or i < start:
            assert not layer.trainable, layer.name


def test_mobilenetv2_preprocessing_matches_keras():
    from plantdx.models.mobilenet import build_mobilenetv2

    model = build_mobilenetv2(image_size=32, num_classes=3, weights=None)
    x = np.array([[[[0.0, 127.5, 255.0]]]], np.float32)
    ours = model.get_layer("preprocess")(x).numpy()
    ref = keras.applications.mobilenet_v2.preprocess_input(x.copy())
    np.testing.assert_allclose(ours, ref, atol=1e-6)


def test_mobilenetv2_unknown_fine_tune_layer():
    from plantdx.models.mobilenet import build_mobilenetv2

    with pytest.raises(ValueError):
        build_mobilenetv2(image_size=32, weights=None, fine_tune_from="block_99")


def test_stage4_configs_chain():
    head = load_config(CONFIGS / "mobilenetv2_head.yaml")
    ft = load_config(CONFIGS / "mobilenetv2_ft.yaml")
    ft3 = load_config(CONFIGS / "mobilenetv2_ft_lr3e-5.yaml")
    assert head["model"]["fine_tune_from"] is None and head["train"]["learning_rate"] == 1e-3
    assert ft["model"]["fine_tune_from"] == "block_13_expand"
    assert ft["train"]["init_from"] == head["run_id"] == "mobilenetv2_head"
    assert ft["train"]["learning_rate"] == 1e-5 and ft3["train"]["learning_rate"] == 3e-5
    a = load_config(CONFIGS / "mobilenetv2_ft_lr1e-4.yaml")
    b = load_config(CONFIGS / "mobilenetv2_ft_all.yaml")
    assert a["model"]["fine_tune_from"] == "block_13_expand" and a["train"]["learning_rate"] == 1e-4
    assert b["model"]["fine_tune_from"] == "Conv1" and b["train"]["learning_rate"] == 3e-5
    for cfg in (a, b):
        assert cfg["train"]["init_from"] == "mobilenetv2_head" and cfg["train"]["epochs"] == 30
    ids = {c["run_id"] for c in (head, ft, ft3, a, b)}
    assert len(ids) == 5


def test_whole_backbone_fine_tune_keeps_batchnorm_frozen():
    from plantdx.models.mobilenet import BACKBONE, build_mobilenetv2

    model = build_mobilenetv2(weights=None, fine_tune_from="Conv1")
    assert sum(int(np.prod(w.shape)) for w in model.trainable_weights) == 2_238_438
    for layer in model.get_layer(BACKBONE).layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            assert not layer.trainable


def test_fine_tune_starts_from_head_checkpoint(tmp_path, monkeypatch):
    from plantdx.models.mobilenet import build_mobilenetv2
    from plantdx.train import load_initial_weights

    monkeypatch.setenv("PLANTDX_ARTIFACTS", str(tmp_path))
    head = build_mobilenetv2(image_size=32, num_classes=4, weights=None)
    for w in head.weights:
        w.assign(np.random.default_rng(1).normal(size=w.shape).astype(w.dtype))
    (tmp_path / "runs" / "head").mkdir(parents=True)
    head.save(tmp_path / "runs" / "head" / "best.keras")

    ft = build_mobilenetv2(
        image_size=32, num_classes=4, weights=None, fine_tune_from="block_13_expand"
    )
    assert load_initial_weights(ft, "head") == len(ft.weights)
    for a, b in zip(head.weights, ft.weights, strict=True):
        assert a.path == b.path
        np.testing.assert_array_equal(a.numpy(), b.numpy())


# ------------------------------------------------------------- selection + test (4.3-4.4)
def test_evaluate_scores_test_once_and_updates_size(tmp_path, monkeypatch):
    from plantdx.evaluate import evaluate_run
    from plantdx.train import train

    monkeypatch.setenv("PLANTDX_ARTIFACTS", str(tmp_path / "artifacts"))
    cfg, data, splits = make_fake_project(tmp_path)
    train(cfg, data, splits_dir=splits, verbose=0)
    run = tmp_path / "artifacts" / "runs" / "smoke"
    (run / "inference.keras").unlink()  # like a run trained before D-014

    report = evaluate_run("smoke", data, splits_dir=splits)
    assert (run / "inference.keras").is_file() and (run / "metrics_test.json").is_file()
    assert report["test"]["n"] == 8 and report["cpu_ms"] > 0
    # the fake test split is all "map": the verified part equals the full split
    assert report["test_verified"]["n"] == 8

    table = pd.read_csv(tmp_path / "artifacts" / "reports" / "results.csv")
    assert sorted(table["split"]) == ["test", "test_verified", "val"]
    assert table["size_mb"].nunique() == 1 and table["cpu_ms"].notna().all()

    with pytest.raises(RuntimeError, match="used once"):
        evaluate_run("smoke", data, splits_dir=splits)


def test_select_and_evaluate_resumes_without_repeating_test(tmp_path, monkeypatch, capsys):
    from plantdx.evaluate import select_and_evaluate
    from plantdx.train import train

    monkeypatch.setenv("PLANTDX_ARTIFACTS", str(tmp_path / "artifacts"))
    cfg, data, splits = make_fake_project(tmp_path)
    train(cfg, data, splits_dir=splits, verbose=0)
    metrics = tmp_path / "artifacts" / "runs" / "smoke" / "metrics_test.json"

    final, _ = select_and_evaluate(data, ["smoke"], splits_dir=splits)
    assert final == "smoke" and metrics.is_file()
    first = metrics.read_text()

    # an interrupted session re-runs the notebook: the existing evaluation is kept, not repeated
    final, table = select_and_evaluate(data, ["smoke"], splits_dir=splits)
    assert "already evaluated on test, kept" in capsys.readouterr().out
    assert metrics.read_text() == first
    assert sorted(table["split"]) == ["test", "test_verified", "val"]
