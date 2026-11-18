"""Run the classical baselines end to end (plan steps 3.1-3.2).

    python -m plantdx.baselines.run --data-root /content/data

1. Hand-crafted features for train and val (cached as .npz under <artifacts>/features).
2. For each model (logreg, linsvm): C grid on train, selection by val macro-F1.
3. Per model, in <artifacts>/runs/<run_id>/: model.joblib, metrics_val.json (with per-class
   scores), grid.csv; one row per model in the shared results table.

The test split is not used here: like every other model, the selected baselines are scored on
test once, in the final evaluation (plan step 4.4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from plantdx.baselines.classical import DEFAULT_CS, DEFAULT_PCA, MODELS, fit_grid, scores
from plantdx.baselines.features import FEATURE_VERSION, feature_dim, split_features
from plantdx.config import load_config, set_seed
from plantdx.data.pipeline import load_split
from plantdx.data.source import CLASSES
from plantdx.metrics import per_class_metrics
from plantdx.paths import artifacts_dir, repo_root, run_dir
from plantdx.results import make_row, results_path, upsert_rows


def run(
    data_root: str | Path,
    config: str | Path | None = None,
    models: tuple[str, ...] = MODELS,
    Cs: tuple[float, ...] = DEFAULT_CS,
    size: int = 128,
    pca: int | None = DEFAULT_PCA,
    splits_dir: str | Path | None = None,
    n_jobs: int = -1,
) -> pd.DataFrame:
    import joblib

    cfg = load_config(config or repo_root() / "configs" / "base.yaml")
    seed = cfg.get("seed", 42)
    set_seed(seed)
    variant = cfg.get("data", {}).get("variant", "color")
    cache = artifacts_dir() / "features"

    data = {}
    for split in ("train", "val"):
        frame = load_split(split, splits_dir)
        data[split] = split_features(
            frame, data_root, variant, size=size, cache_dir=cache, tag=split, n_jobs=n_jobs
        )
        print(f"{split}: X {data[split][0].shape}")
    (X_tr, y_tr), (X_va, y_va) = data["train"], data["val"]

    rows = []
    for name in models:
        res = fit_grid(name, X_tr, y_tr, X_va, y_va, Cs=Cs, seed=seed, pca=pca, feature_size=size)
        run_id = f"baseline_{name}"
        out = run_dir(run_id)
        joblib.dump(res.best_model, out / "model.joblib")
        pd.DataFrame(res.trials).to_csv(out / "grid.csv", index=False)

        y_pred = scores(res.best_model, X_va).argmax(axis=1)
        hyper = {
            "C": res.best_C,
            "C_grid": list(Cs),
            "features": f"HSV 3x32 + HOG 9x16px + LBP u8,1 @ {size}px (v{FEATURE_VERSION})",
            "feature_dim": feature_dim(size),
            "hog_pca": pca,
            "class_weight": "balanced",
        }
        metrics = {k: v for k, v in res.best_val.items() if k != "train_seconds"}
        report = {
            "run_id": run_id,
            "model": name,
            "split": "val",
            "metrics": metrics,
            "hyperparams": hyper,
            "train_seconds": res.best_val["train_seconds"],
            "per_class": per_class_metrics(y_va, y_pred, CLASSES),
        }
        (out / "metrics_val.json").write_text(json.dumps(report, indent=2) + "\n")
        size_mb = (out / "model.joblib").stat().st_size / 1e6
        rows.append(
            make_row(
                run_id,
                name,
                "val",
                metrics,
                hyper,
                size_mb=round(size_mb, 2),
                train_seconds=round(res.best_val["train_seconds"], 1),
            )
        )
        print(f"-> {run_id}: C={res.best_C}, val macro-F1 {metrics['macro_f1']:.4f}\n")

    table = upsert_rows(rows, results_path(cfg))
    print(f"results: {results_path(cfg)}")
    return table


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True, help="folder that contains raw/<variant>/")
    parser.add_argument("--config", default=str(repo_root() / "configs" / "base.yaml"))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--C", nargs="+", type=float, default=list(DEFAULT_CS))
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--pca", type=int, default=DEFAULT_PCA, help="0 = no PCA")
    parser.add_argument("--splits-dir", default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args(argv)
    run(
        args.data_root,
        args.config,
        tuple(args.models),
        tuple(args.C),
        args.size,
        args.pca or None,
        args.splits_dir,
        args.n_jobs,
    )


if __name__ == "__main__":
    main()
