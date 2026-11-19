"""Error analysis on saved predictions (plan steps 5.1-5.5): no TensorFlow needed."""

import numpy as np
import pandas as pd
import pytest

from plantdx import analysis as A


def test_confusion_and_row_normalisation():
    y = np.array([0, 0, 1, 1, 1, 2])
    p = np.array([0, 1, 1, 1, 0, 2])
    cm = A.confusion(y, p, 4)
    assert cm.tolist() == [[1, 1, 0, 0], [1, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
    norm = A.normalize_rows(cm)
    assert norm[0].tolist() == [0.5, 0.5, 0, 0]
    assert np.allclose(norm[1], [1 / 3, 2 / 3, 0, 0])
    assert norm[3].sum() == 0  # empty row stays zero


def test_top_confusions_orders_off_diagonal_cells():
    cm = np.array([[10, 3, 0], [1, 8, 0], [0, 5, 5]])
    top = A.top_confusions(cm, ["a", "b", "c"], k=5)
    assert top[["true", "predicted", "count"]].values.tolist() == [
        ["c", "b", 5],
        ["a", "b", 3],
        ["b", "a", 1],
    ]
    assert top["share_of_true"].iloc[0] == pytest.approx(0.5)


def test_weakest_classes_skips_absent_ones():
    rows = [
        {"class": "a", "support": 10, "precision": 1, "recall": 0.5, "f1": 0.67},
        {"class": "b", "support": 0, "precision": None, "recall": None, "f1": None},
        {"class": "c", "support": 5, "precision": 1, "recall": 1, "f1": 1.0},
    ]
    assert A.weakest_classes(rows, k=5)["class"].tolist() == ["a", "c"]


def probs_for(y, conf, pred):
    """Probability rows with the given top class and top probability (3 classes)."""
    p = np.full((len(y), 3), 0.0)
    for i, (c, k) in enumerate(zip(conf, pred, strict=True)):
        p[i] = (1 - c) / 2
        p[i, k] = c
    return p


def test_threshold_rule_keeps_the_most_answers_that_reach_the_target():
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    # 8 confident correct answers, 2 errors at moderate confidence
    conf = np.array([0.99, 0.98, 0.97, 0.96, 0.95, 0.99, 0.999, 0.97, 0.6, 0.7])
    pred = np.array([0, 1, 2, 0, 1, 2, 0, 1, 0, 1])
    p = probs_for(y, conf, pred)
    t = A.choose_threshold(p, y, target=0.995)
    assert 0.7 < t <= 0.95  # rejects both errors, keeps every correct answer
    res = A.apply_threshold(p, y, t)
    assert res["coverage"] == pytest.approx(0.8)
    assert res["accuracy_answered"] == 1.0 and res["accuracy_all"] == pytest.approx(0.8)
    assert res["errors_caught"] == 2 and res["share_errors_caught"] == 1.0
    assert res["correct_rejected"] == 0


def test_threshold_when_target_cannot_be_reached():
    y = np.array([0, 1])
    p = probs_for(y, [0.9, 0.95], [1, 0])  # both wrong
    t = A.choose_threshold(p, y, target=0.995)
    assert 0 <= t <= 1  # falls back to the best available threshold, no error


def test_coverage_curve_is_monotone():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(4), size=200)
    y = rng.integers(0, 4, 200)
    curve = A.coverage_curve(p, y)
    assert curve["coverage"].iloc[0] == 1.0
    assert (np.diff(curve["coverage"]) <= 1e-12).all()
    assert (curve["errors_answered"] + curve["errors_rejected"]).nunique() == 1


def test_default_thresholds_are_finest_near_one():
    t = A.default_thresholds()
    assert t[0] == 0 and t[-1] == 1 and (np.diff(t) > 0).all()
    assert np.diff(t[t >= 0.99]).max() <= 1e-4 + 1e-12


def test_failure_cases_prefer_top_confusions_then_most_confident():
    frame = pd.DataFrame(
        {
            "path": [f"img{i}" for i in range(6)],
            "label": list("aabbcc"),
            "label_index": [0, 0, 1, 1, 2, 2],
        }
    )
    y = frame["label_index"].to_numpy()
    pred = np.array(
        [1, 0, 0, 0, 1, 2]
    )  # errors: img0 (a->b), img2 (b->a), img3 (b->a), img4 (c->b)
    conf = np.array([0.9, 0.99, 0.6, 0.8, 0.95, 0.99])
    p = probs_for(y, conf, pred)
    cm = A.confusion(y, pred, 3)
    top = A.top_confusions(cm, ["a", "b", "c"], k=1)  # b->a (2 errors)
    cases = A.failure_cases(p, frame, top, n=3)
    assert cases["path"].tolist() == ["img3", "img4", "img0"]  # b->a best, then by confidence
    assert (cases["predicted_index"] != cases["label_index"]).all()


def test_leak_pairs_only_cross_train_test():
    pairs = pd.DataFrame(
        {
            "path_a": ["t1", "x1", "t2", "x3"],
            "path_b": ["x2", "t3", "t4", "x4"],
            "corr_hi": [0.93, 0.95, 0.99, 0.97],
            "transform": ["identity"] * 4,
        }
    )
    leaks = A.leak_pairs(pairs, {"t1", "t2", "t3", "t4"}, k=5)
    assert leaks["corr_hi"].tolist() == [0.95, 0.93]  # both-in-test and no-test pairs excluded
