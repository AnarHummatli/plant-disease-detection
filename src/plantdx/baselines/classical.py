"""Linear classifiers on hand-crafted features (plan step 3.2).

    HOG -> StandardScaler -> PCA(256)  ┐
    HSV + LBP histograms (106 values)  ┴-> StandardScaler -> LogisticRegression | LinearSVC

with class_weight="balanced".

The regularisation strength C is chosen on the validation split by macro-F1 (the project's
selection metric); the model is fitted on train only. The test split is never touched here.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from plantdx.baselines.features import feature_slices
from plantdx.metrics import summary_metrics

MODELS = ("logreg", "linsvm")
DEFAULT_CS = (0.01, 0.1, 1.0, 10.0)
# HOG is 1,764 of the 1,870 features and strongly correlated; reducing it to 256 PCA components
# keeps the linear models fast enough for a C grid on Colab CPUs (D-011).
DEFAULT_PCA = 256


def make_model(
    name: str,
    C: float,
    seed: int = 42,
    max_iter: int | None = None,
    pca: int | None = DEFAULT_PCA,
    feature_size: int = 128,
) -> Pipeline:
    """[HOG -> StandardScaler -> PCA(pca)] + colour/LBP -> StandardScaler -> classifier.

    `pca=None` uses all features (StandardScaler -> classifier). `feature_size` is the image
    size the features were computed at; it fixes where the HOG block sits in the vector.
    """
    if name == "logreg":
        clf = LogisticRegression(
            C=C, class_weight="balanced", max_iter=max_iter or 1000, random_state=seed
        )
    elif name == "linsvm":
        # primal solver: far more samples than features
        clf = LinearSVC(
            C=C, class_weight="balanced", dual=False, max_iter=max_iter or 5000, random_state=seed
        )
    else:
        raise ValueError(f"unknown model {name!r}; expected one of {MODELS}")
    if not pca:
        return make_pipeline(StandardScaler(), clf)
    # PCA only on the HOG block: it holds 1,764 of the 1,870 values, so a PCA over everything
    # would be dominated by it and could drop the colour and LBP histograms, which carry most
    # of the symptom colour. Those stay as they are.
    sl = feature_slices(feature_size)
    hog_cols = list(range(sl["hog"].start, sl["hog"].stop))
    other_cols = list(range(sl["hsv"].start, sl["hsv"].stop)) + list(
        range(sl["lbp"].start, sl["lbp"].stop)
    )
    reduce = ColumnTransformer(
        [
            (
                "hog",
                make_pipeline(
                    StandardScaler(),
                    PCA(n_components=pca, svd_solver="randomized", random_state=seed),
                ),
                hog_cols,
            ),
            ("colour_lbp", "passthrough", other_cols),
        ]
    )
    return make_pipeline(reduce, StandardScaler(), clf)


def scores(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Per-class scores: probabilities if the model has them, else decision values."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)
        except AttributeError:  # LinearSVC exposes the name only through the pipeline
            pass
    return model.decision_function(X)


@dataclass
class GridResult:
    name: str
    best_C: float
    best_model: Pipeline
    best_val: dict[str, float]
    trials: list[dict[str, Any]] = field(default_factory=list)


def fit_grid(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    Cs: tuple[float, ...] | list[float] = DEFAULT_CS,
    seed: int = 42,
    max_iter: int | None = None,
    pca: int | None = DEFAULT_PCA,
    feature_size: int = 128,
    verbose: bool = True,
) -> GridResult:
    """Fit one model per C on train, score on val, keep the best by macro-F1.

    Ties are broken by the smaller C (the stronger regularisation, i.e. the simpler model).
    """
    trials, best = [], None
    for C in sorted(Cs):
        model = make_model(
            name, C, seed=seed, max_iter=max_iter, pca=pca, feature_size=feature_size
        )
        t0 = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X_train, y_train)
        seconds = time.perf_counter() - t0
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        val = summary_metrics(y_val, scores(model, X_val))
        trial = {
            "model": name,
            "C": C,
            "pca": pca,
            "fit_seconds": round(seconds, 1),
            "converged": converged,
        }
        trial.update({f"val_{k}": v for k, v in val.items()})
        trials.append(trial)
        if verbose:
            note = "" if converged else ", not converged"
            print(
                f"{name:7s} C={C:<6g} val macro-F1 {val['macro_f1']:.4f} "
                f"acc {val['accuracy']:.4f}  ({seconds:.0f} s{note})"
            )
        if best is None or val["macro_f1"] > best[2]["macro_f1"]:
            best = (C, model, val, seconds)
    C, model, val, seconds = best
    result = GridResult(name=name, best_C=C, best_model=model, best_val=val, trials=trials)
    result.best_val["train_seconds"] = seconds
    return result
