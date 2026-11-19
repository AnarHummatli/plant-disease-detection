"""Error analysis of the final model (plan steps 5.1-5.5, D-019).

Everything here works on saved predictions (val_scores.npz from training, test_scores.npz from
the one-time test evaluation), so the analysis never scores the test split again.

* confusion matrix (counts and row-normalised), most-confused class pairs, weakest classes;
* "not sure" threshold: chosen on **validation** by a rule fixed before looking at the result
  (smallest confidence threshold whose answered images reach the target accuracy), then applied
  unchanged to test;
* the cross-split near-identical pairs that the duplicate search removed (D-006), for a figure.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

TARGET_ACCURACY = 0.995  # D-019: answered images must be at least 99.5% correct on val


def confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Counts [true, predicted]."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def normalize_rows(cm: np.ndarray) -> np.ndarray:
    """Each row divided by its total (recall on the diagonal); empty rows stay zero."""
    totals = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, totals, out=np.zeros(cm.shape, dtype=float), where=totals > 0)


def top_confusions(cm: np.ndarray, classes: Sequence[str], k: int = 5) -> pd.DataFrame:
    """The k largest off-diagonal cells: true and predicted class, count, share of true class."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0]
    rows = []
    for t, p in order[:k]:
        if off[t, p] == 0:
            break
        rows.append(
            {
                "true": classes[t],
                "predicted": classes[p],
                "count": int(off[t, p]),
                "share_of_true": float(off[t, p] / cm[t].sum()),
                "true_index": int(t),
                "predicted_index": int(p),
            }
        )
    return pd.DataFrame(rows)


def weakest_classes(per_class: list[dict], k: int = 5) -> pd.DataFrame:
    table = pd.DataFrame([r for r in per_class if r.get("f1") is not None])
    return table.sort_values(["f1", "support"]).head(k).reset_index(drop=True)


def default_thresholds() -> np.ndarray:
    """0-0.9 in steps of 0.01, 0.9-0.99 in 0.001, 0.99-1 in 0.0001: a confident model keeps most
    of its probabilities above 0.99, so the grid is finest there."""
    grid = np.concatenate(
        [
            np.arange(0.0, 0.9, 0.01),
            np.arange(0.9, 0.99, 0.001),
            np.arange(0.99, 1.0, 0.0001),
            [1.0],
        ]
    )
    return np.unique(np.round(grid, 4))


def coverage_curve(probs: np.ndarray, y: np.ndarray, thresholds=None) -> pd.DataFrame:
    """For each threshold t: share of images answered (max prob >= t) and their accuracy."""
    probs, y = np.asarray(probs, dtype=np.float64), np.asarray(y)
    conf, correct = probs.max(axis=1), probs.argmax(axis=1) == y
    if thresholds is None:
        thresholds = default_thresholds()
    rows = []
    for t in thresholds:
        answered = conf >= t
        n = int(answered.sum())
        rows.append(
            {
                "threshold": float(t),
                "coverage": n / len(y),
                "accuracy_answered": float(correct[answered].mean()) if n else float("nan"),
                "errors_answered": int((~correct & answered).sum()),
                "errors_rejected": int((~correct & ~answered).sum()),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(
    probs_val: np.ndarray, y_val: np.ndarray, target: float = TARGET_ACCURACY
) -> float:
    """Smallest threshold whose answered validation images reach `target` accuracy.

    The smallest such threshold keeps the most images answered. If the target is never reached,
    the threshold with the best answered accuracy is returned.
    """
    curve = coverage_curve(probs_val, y_val)
    ok = curve[curve["accuracy_answered"] >= target]
    if len(ok):
        return float(ok["threshold"].min())
    return float(curve.loc[curve["accuracy_answered"].idxmax(), "threshold"])


def apply_threshold(probs: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float]:
    probs, y = np.asarray(probs, dtype=np.float64), np.asarray(y)
    conf, correct = probs.max(axis=1), probs.argmax(axis=1) == y
    answered = conf >= threshold
    errors = int((~correct).sum())
    return {
        "threshold": float(threshold),
        "n": int(len(y)),
        "coverage": float(answered.mean()),
        "accuracy_all": float(correct.mean()),
        "accuracy_answered": float(correct[answered].mean()) if answered.any() else float("nan"),
        "errors_total": errors,
        "errors_caught": int((~correct & ~answered).sum()),
        "share_errors_caught": float((~correct & ~answered).sum() / errors) if errors else 0.0,
        "correct_rejected": int((correct & ~answered).sum()),
    }


def failure_cases(
    probs: np.ndarray, frame: pd.DataFrame, confusions: pd.DataFrame, n: int = 6
) -> pd.DataFrame:
    """Up to n misclassified images: the most confident error of each top confusion pair, then
    the most confident remaining errors, one image per (true, predicted) pair."""
    probs = np.asarray(probs, dtype=np.float64)
    y = frame["label_index"].to_numpy()
    pred, conf = probs.argmax(axis=1), probs.max(axis=1)
    wrong = np.flatnonzero(pred != y)
    wrong = wrong[np.argsort(-conf[wrong])]
    chosen, pairs = [], set()
    for _, row in confusions.iterrows():
        pair = (row["true_index"], row["predicted_index"])
        hits = [i for i in wrong if (y[i], pred[i]) == pair]
        if hits and len(chosen) < n:
            chosen.append(hits[0])
            pairs.add(pair)
    for i in wrong:
        if len(chosen) >= n:
            break
        if (y[i], pred[i]) not in pairs:
            chosen.append(i)
            pairs.add((y[i], pred[i]))
    out = frame.iloc[chosen][["path", "label", "label_index"]].copy()
    out["predicted_index"] = pred[chosen]
    out["confidence"] = conf[chosen]
    out["true_probability"] = probs[chosen, y[chosen]]
    out["index"] = chosen
    return out.reset_index(drop=True)


def leak_pairs(duplicate_pairs: pd.DataFrame, test_paths: set[str], k: int = 8) -> pd.DataFrame:
    """Near-identical pairs that crossed train/test in the official split (one image in test),
    most similar first: the leakage the duplicate search removed (D-006)."""
    a_test = duplicate_pairs["path_a"].isin(test_paths)
    b_test = duplicate_pairs["path_b"].isin(test_paths)
    crossing = duplicate_pairs[a_test ^ b_test].copy()
    return crossing.sort_values("corr_hi", ascending=False).head(k).reset_index(drop=True)
