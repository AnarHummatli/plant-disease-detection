"""Final model selection and the one-time test evaluation (plan steps 4.3-4.4, D-015).

    python -m plantdx.evaluate --data-root /content/data \\
        --select mobilenetv2_head mobilenetv2_ft mobilenetv2_ft_lr3e-5 \\
        --also baseline_logreg baseline_linsvm cnn_scratch

1. **Selection (4.3)** among the `--select` runs, on validation only: best val macro-F1, ties to
   the smaller model. Written to <artifacts>/reports/final_model.json.
2. **Test (4.4)** for the selected model and for the `--also` runs (the model already chosen in
   every other family, for the comparison table). Every run is scored on
   * the full test split (10,709 images, 38 classes) and
   * its leaf-verified part (leaf_source == "map": 8,033 images, 30 classes), where leaf
     grouping rules out same-leaf leakage by construction (D-004);
   the difference between the two is the effect of any residual leakage.
3. Per run: metrics_test.json (both parts, per-class scores), test_scores.npz, CPU latency
   (median ms per image, batch 1), and rows "test" / "test_verified" in the results table.

The test split is evaluated **once**: a run that already has metrics_test.json is refused
unless --force is given, and nothing is selected or tuned after this step.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml

from plantdx.data.pipeline import load_split
from plantdx.data.source import CLASSES
from plantdx.metrics import per_class_metrics, summary_metrics
from plantdx.paths import artifacts_dir, run_dir
from plantdx.results import make_row, results_path, select_final, update_run_fields, upsert_rows

VERIFIED = "map"


def is_classical(run_id: str) -> bool:
    return (run_dir(run_id, create=False) / "model.joblib").is_file()


def run_config(run_id: str) -> dict[str, Any]:
    path = run_dir(run_id, create=False) / "config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}


def median_ms(fn, n: int = 50, warmup: int = 5) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


# --------------------------------------------------------------------------- neural models
def keras_scores(run_id: str, frame: pd.DataFrame, data_root, splits_dir=None):
    """(test probabilities, CPU ms per image, size MB, params) from inference.keras."""
    import keras
    import tensorflow as tf

    from plantdx.data.pipeline import make_dataset
    from plantdx.export import INFERENCE_FILE, export_run

    folder = run_dir(run_id, create=False)
    path = folder / INFERENCE_FILE
    if not path.is_file():  # runs trained before D-014
        export_run(run_id)
    cfg = run_config(run_id)
    data = cfg.get("data", {})
    size = data.get("image_size", 224)

    model = keras.models.load_model(path)
    ds = make_dataset(
        frame,
        data_root,
        variant=data.get("variant", "color"),
        image_size=size,
        batch_size=cfg.get("train", {}).get("batch_size", 64),
        num_classes=data.get("num_classes", len(CLASSES)),
    )
    probs = np.asarray(model.predict(ds, verbose=0), dtype=np.float32)

    with tf.device("/CPU:0"):
        cpu_model = keras.models.load_model(path)
        x = tf.constant(np.random.default_rng(0).uniform(0, 255, (1, size, size, 3)), tf.float32)
        ms = median_ms(lambda: cpu_model(x, training=False))
    return probs, ms, path.stat().st_size / 1e6, int(model.count_params())


# ------------------------------------------------------------------------ classical models
def classical_scores(run_id: str, frame: pd.DataFrame, data_root):
    """(test scores, CPU ms per image incl. feature extraction, size MB, None)."""
    import joblib

    from plantdx.baselines.classical import scores
    from plantdx.baselines.features import image_features, load_rgb, split_features
    from plantdx.data.images import image_path

    folder = run_dir(run_id, create=False)
    report = json.loads((folder / "metrics_val.json").read_text())
    feature_size = int(report.get("hyperparams", {}).get("feature_size", 128))
    model = joblib.load(folder / "model.joblib")
    X, _ = split_features(
        frame, data_root, size=feature_size, cache_dir=artifacts_dir() / "features", tag="test"
    )
    s = scores(model, X)

    one = str(image_path(data_root, "color", frame["path"].iloc[0]))
    ms = median_ms(lambda: scores(model, image_features(load_rgb(one, feature_size))[None]))
    return s, ms, (folder / "model.joblib").stat().st_size / 1e6, None


# -------------------------------------------------------------------------------- driver
def evaluate_run(
    run_id: str, data_root, splits_dir=None, force: bool = False, cfg=None
) -> dict[str, Any]:
    folder = run_dir(run_id, create=False)
    out = folder / "metrics_test.json"
    if out.exists() and not force:
        raise RuntimeError(
            f"{run_id} was already evaluated on test ({out}); the test split is used once. "
            "Pass --force only to repeat an identical evaluation that failed midway."
        )
    frame = load_split("test", splits_dir)
    if is_classical(run_id):
        scores_, ms, size_mb, params = classical_scores(run_id, frame, data_root)
    else:
        scores_, ms, size_mb, params = keras_scores(run_id, frame, data_root, splits_dir)

    y = frame["label_index"].to_numpy()
    verified = (frame["leaf_source"] == VERIFIED).to_numpy()
    pred = scores_.argmax(axis=1)
    report = {
        "run_id": run_id,
        "test": summary_metrics(y, scores_),
        "test_verified": summary_metrics(y[verified], scores_[verified]),
        "cpu_ms": round(ms, 2),
        "size_mb": round(size_mb, 2),
        "params": params,
        "per_class": per_class_metrics(y, pred, CLASSES),
        "per_class_verified": per_class_metrics(y[verified], pred[verified], CLASSES),
    }
    out.write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(
        folder / "test_scores.npz", scores=scores_.astype(np.float16), y=y, verified=verified
    )

    table_path = results_path(cfg)
    val = pd.read_csv(table_path)
    val = val[(val["run_id"] == run_id) & (val["split"] == "val")]
    model_name = str(val["model"].iloc[0]) if len(val) else run_id
    hyper = json.loads(val["hyperparams"].iloc[0]) if len(val) else {}
    rows = [
        make_row(
            run_id,
            model_name,
            split,
            report[split],
            hyper,
            params=params,
            size_mb=report["size_mb"],
            cpu_ms=report["cpu_ms"],
        )
        for split in ("test", "test_verified")
    ]
    upsert_rows(rows, table_path)
    fields = {"size_mb": report["size_mb"], "cpu_ms": report["cpu_ms"]}
    if params is not None:
        fields["params"] = params
    update_run_fields(table_path, run_id, **fields)
    t, v = report["test"], report["test_verified"]
    print(
        f"{run_id:24s} test macro-F1 {t['macro_f1']:.4f} acc {t['accuracy']:.4f} | "
        f"verified macro-F1 {v['macro_f1']:.4f} acc {v['accuracy']:.4f} | "
        f"{report['size_mb']:.2f} MB, {report['cpu_ms']:.1f} ms/img (CPU)"
    )
    return report


def select_and_evaluate(
    data_root,
    select: list[str],
    also: list[str] | None = None,
    splits_dir=None,
    force: bool = False,
) -> tuple[str, pd.DataFrame]:
    table = pd.read_csv(results_path())
    present = [r for r in select if ((table["run_id"] == r) & (table["split"] == "val")).any()]
    missing = sorted(set(select) - set(present))
    if missing:
        print(f"not trained, skipped from selection: {missing}")
    final, ranked = select_final(table, present)
    print("selection on validation (macro-F1, ties -> smaller model):")
    print(ranked[["run_id", "macro_f1", "accuracy", "top3", "size_mb"]].to_string(index=False))
    print(f"-> final model: {final}\n")
    (results_path().parent / "final_model.json").write_text(
        json.dumps(
            {
                "run_id": final,
                "rule": "highest val macro-F1; ties -> smaller model (plan 4.3)",
                "candidates": ranked[["run_id", "macro_f1", "accuracy", "size_mb"]].to_dict(
                    "records"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    for run_id in [final, *(also or [])]:
        evaluate_run(run_id, data_root, splits_dir, force=force)
    return final, pd.read_csv(results_path())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--select", nargs="+", required=True, help="final-model candidates")
    parser.add_argument("--also", nargs="*", default=[], help="other runs to score on test")
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    select_and_evaluate(args.data_root, args.select, args.also, args.splits_dir, args.force)


if __name__ == "__main__":
    main()
