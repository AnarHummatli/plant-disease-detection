"""Classification metrics shared by every model (plan steps 3.2, 3.3a, 5.1).

All functions take integer labels and a score matrix [n, num_classes]. Scores can be
probabilities (softmax) or any ranking score (e.g. LinearSVC's decision_function): accuracy,
F1 and top-k only use the ranking of the classes per row.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


def top_k_accuracy(y_true: np.ndarray, scores: np.ndarray, k: int = 3) -> float:
    """Share of rows whose true class is among the k highest scores."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    k = min(k, scores.shape[1])
    top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    return float((top == y_true[:, None]).any(axis=1).mean())


def summary_metrics(
    y_true: np.ndarray, scores: np.ndarray, num_classes: int | None = None
) -> dict[str, float]:
    """accuracy, macro_f1, weighted_f1, top3, n.

    Macro-F1 is averaged over the classes present in `y_true` (sklearn's default), so a subset
    without some classes (the leaf-verified test part has 30 of 38) is not penalised for them.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if scores.ndim != 2 or len(scores) != len(y_true):
        raise ValueError(f"scores must be [n, classes] with n = {len(y_true)}, got {scores.shape}")
    if num_classes is not None and scores.shape[1] != num_classes:
        raise ValueError(f"expected {num_classes} score columns, got {scores.shape[1]}")
    y_pred = scores.argmax(axis=1)
    present = np.unique(y_true)
    return {
        "n": int(len(y_true)),
        "classes": int(len(present)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=present, average="weighted")),
        "top3": top_k_accuracy(y_true, scores, k=3),
    }


def per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str] | tuple[str, ...]
) -> list[dict[str, Any]]:
    """Precision, recall, F1 and support for every class (support 0 → scores are None)."""
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    rows = []
    for i, name in enumerate(class_names):
        has = int(s[i]) > 0
        rows.append(
            {
                "index": i,
                "class": name,
                "support": int(s[i]),
                "precision": float(p[i]) if has else None,
                "recall": float(r[i]) if has else None,
                "f1": float(f[i]) if has else None,
            }
        )
    return rows
