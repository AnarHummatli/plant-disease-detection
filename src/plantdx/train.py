"""Train any Keras model of this project from an experiment config (plan step 3.3a).

    python -m plantdx.train --config configs/cnn.yaml --data-root /content/data

load_config -> set_seed -> datasets_from_config -> build_model(model.name) -> compile -> fit:

* ModelCheckpoint keeps the epoch with the best **val macro-F1** (the selection metric);
* EarlyStopping and ReduceLROnPlateau watch **val loss**, which is much smoother than macro-F1
  on a validation set whose smallest class has 12 images;
* CSVLogger writes history.csv (the learning curves).

After training, the best checkpoint is scored on val with scikit-learn (exact macro-F1, per-class
scores) and one row goes into the shared results table. Everything for one run lands in
<artifacts>/runs/<run_id>/: config.yaml, best.keras (training checkpoint), inference.keras
(float32, no augmentation, no optimizer state; its file size is the reported model size, D-014),
history.csv, metrics_val.json, val_scores.npz (probabilities, kept for threshold selection and
error analysis later).

The test split is not evaluated here (plan rule: test once, final model only, step 4.4).
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from plantdx.config import load_config, set_seed
from plantdx.data.pipeline import (
    augmentation_from_config,
    class_weights,
    datasets_from_config,
    load_split,
)
from plantdx.data.source import CLASSES
from plantdx.env import report as env_report
from plantdx.export import export_inference
from plantdx.metrics import per_class_metrics, summary_metrics
from plantdx.models import build_model
from plantdx.paths import run_dir
from plantdx.results import make_row, results_path, upsert_rows


def make_optimizer(train_cfg: Mapping[str, Any]):
    import keras

    name = train_cfg.get("optimizer", "adam").lower()
    lr = float(train_cfg.get("learning_rate", 1e-3))
    if name == "adam":
        return keras.optimizers.Adam(learning_rate=lr)
    if name == "adamw":
        wd = float(train_cfg.get("weight_decay", 1e-4))
        return keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    if name == "sgd":
        return keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    raise ValueError(f"unknown optimizer {name!r}; expected adam, adamw or sgd")


def compile_model(model, cfg: Mapping[str, Any]):
    import keras

    t = cfg.get("train", {})
    model.compile(
        optimizer=make_optimizer(t),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=float(t.get("label_smoothing", 0.0))
        ),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
            keras.metrics.F1Score(average="macro", name="macro_f1"),
        ],
    )
    return model


def make_callbacks(out: Path, train_cfg: Mapping[str, Any]) -> list:
    import keras

    return [
        keras.callbacks.ModelCheckpoint(
            str(out / "best.keras"), monitor="val_macro_f1", mode="max", save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(train_cfg.get("early_stopping_patience", 5)),
            restore_best_weights=False,  # best.keras (by macro-F1) is reloaded after fit
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(train_cfg.get("reduce_lr_factor", 0.5)),
            patience=int(train_cfg.get("reduce_lr_patience", 2)),
            min_lr=float(train_cfg.get("min_lr", 1e-6)),
        ),
        keras.callbacks.CSVLogger(str(out / "history.csv")),
        keras.callbacks.TerminateOnNaN(),
    ]


def predict_scores(model, dataset, steps: int | None = None) -> np.ndarray:
    return np.asarray(model.predict(dataset, steps=steps, verbose=0), dtype=np.float32)


def train(
    config: str | Path,
    data_root: str | Path,
    run_id: str | None = None,
    splits_dir: str | Path | None = None,
    epochs: int | None = None,
    max_steps: int | None = None,
    verbose: int = 2,
) -> dict[str, Any]:
    """Train one run and return its val report. `max_steps` caps steps per epoch (smoke tests)."""
    import keras

    cfg = load_config(config)
    run_id = run_id or cfg.get("run_id") or Path(config).stem
    seed = int(cfg.get("seed", 42))
    t = cfg.setdefault("train", {})
    if epochs is not None:
        t["epochs"] = epochs
    set_seed(seed)
    if t.get("mixed_precision"):
        keras.mixed_precision.set_global_policy("mixed_float16")

    out = run_dir(run_id)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (out / "env.json").write_text(json.dumps(env_report(), indent=2, default=str) + "\n")

    ds = datasets_from_config(cfg, data_root, splits_dir)
    model = compile_model(build_model(cfg, augmentation_from_config(cfg)), cfg)
    if verbose:
        model.summary(line_length=100)

    cw = class_weights(load_split("train", splits_dir)) if t.get("class_weight") else None
    t0 = time.perf_counter()
    history = model.fit(
        ds["train"],
        validation_data=ds["val"],
        epochs=int(t.get("epochs", 30)),
        steps_per_epoch=max_steps,
        validation_steps=max_steps,
        callbacks=make_callbacks(out, t),
        class_weight=cw,
        verbose=verbose,
    )
    train_seconds = time.perf_counter() - t0

    best_path = out / "best.keras"
    best = keras.models.load_model(best_path) if best_path.exists() else model
    val_frame = load_split("val", splits_dir)
    probs = predict_scores(best, ds["val"], steps=max_steps)
    y_val = val_frame["label_index"].to_numpy()[: len(probs)]
    np.savez_compressed(out / "val_scores.npz", probs=probs.astype(np.float16), y=y_val)

    metrics = summary_metrics(y_val, probs)
    hist = history.history
    best_epoch = int(np.nanargmax(hist["val_macro_f1"])) + 1 if hist.get("val_macro_f1") else None
    inference_path = export_inference(cfg, best, out)
    size_mb = inference_path.stat().st_size / 1e6
    hyper = {
        "model": cfg.get("model", {}),
        "optimizer": t.get("optimizer", "adam"),
        "learning_rate": t.get("learning_rate", 1e-3),
        "batch_size": t.get("batch_size", 64),
        "epochs_max": t.get("epochs", 30),
        "label_smoothing": t.get("label_smoothing", 0.0),
        "class_weight": bool(t.get("class_weight")),
        "augment_on": t.get("augment_on", "model"),
        "augment": cfg.get("augment", {}),
        "image_size": cfg.get("data", {}).get("image_size", 224),
        "mixed_precision": bool(t.get("mixed_precision")),
    }
    report = {
        "run_id": run_id,
        "model": model.name,
        "split": "val",
        "metrics": metrics,
        "params": int(best.count_params()),
        "size_mb": round(size_mb, 2),
        "epochs_trained": len(hist.get("loss", [])),
        "best_epoch": best_epoch,
        "train_seconds": round(train_seconds, 1),
        "hyperparams": hyper,
        "per_class": per_class_metrics(y_val, probs.argmax(1), CLASSES),
    }
    (out / "metrics_val.json").write_text(json.dumps(report, indent=2) + "\n")

    row = make_row(
        run_id,
        model.name,
        "val",
        metrics,
        hyper,
        params=report["params"],
        size_mb=report["size_mb"],
        train_seconds=report["train_seconds"],
        epochs=report["epochs_trained"],
    )
    upsert_rows([row], results_path(cfg))
    if verbose:
        m = metrics
        print(
            f"\n{run_id}: val macro-F1 {m['macro_f1']:.4f} · accuracy {m['accuracy']:.4f} · "
            f"top-3 {m['top3']:.4f} · best epoch {best_epoch}/{report['epochs_trained']} · "
            f"{train_seconds / 60:.1f} min\nrun folder: {out}"
        )
    keras.mixed_precision.set_global_policy("float32")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True, help="folder that contains raw/<variant>/")
    parser.add_argument("--run-id", default=None, help="default: config file name")
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="steps per epoch (smoke)")
    args = parser.parse_args(argv)
    train(
        args.config,
        args.data_root,
        run_id=args.run_id,
        splits_dir=args.splits_dir,
        epochs=args.epochs,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
