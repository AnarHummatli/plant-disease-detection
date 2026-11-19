"""Shared metrics and the results table (plan steps 3.2, 3.3a, 5.1)."""

import json

import numpy as np
import pandas as pd
import pytest

from plantdx.metrics import per_class_metrics, summary_metrics, top_k_accuracy
from plantdx.results import COLUMNS, make_row, resolve_artifact, upsert_rows


def test_top_k_accuracy():
    scores = np.array([[0.1, 0.5, 0.4], [0.7, 0.2, 0.1], [0.3, 0.3, 0.4]])
    y = np.array([2, 2, 0])
    assert top_k_accuracy(y, scores, k=1) == pytest.approx(0.0)
    assert top_k_accuracy(y, scores, k=2) == pytest.approx(2 / 3)
    assert top_k_accuracy(y, scores, k=3) == 1.0
    assert top_k_accuracy(y, scores, k=10) == 1.0  # k larger than the number of classes


def test_summary_metrics_perfect_and_values():
    y = np.array([0, 1, 2, 2])
    perfect = np.eye(3)[y]
    m = summary_metrics(y, perfect)
    assert m["accuracy"] == m["macro_f1"] == m["weighted_f1"] == m["top3"] == 1.0
    assert (m["n"], m["classes"]) == (4, 3)

    wrong = np.eye(3)[[0, 1, 2, 0]]  # last image of class 2 predicted as 0
    m = summary_metrics(y, wrong)
    assert m["accuracy"] == pytest.approx(0.75)
    # F1 per class: 0 -> 2/3, 1 -> 1, 2 -> 2/3
    assert m["macro_f1"] == pytest.approx((2 / 3 + 1 + 2 / 3) / 3)


def test_macro_f1_ignores_classes_absent_from_y_true():
    # the leaf-verified test subset has 30 of 38 classes: absent classes must not count as 0
    y = np.array([0, 0, 1, 1])
    scores = np.zeros((4, 5))
    scores[np.arange(4), y] = 1
    m = summary_metrics(y, scores, num_classes=5)
    assert m["macro_f1"] == 1.0 and m["classes"] == 2


def test_decision_scores_work_like_probabilities():
    y = np.array([0, 1, 2])
    svm_like = np.array([[2.0, -1.0, -3.0], [-0.5, 0.3, -2.0], [-1.0, -2.0, 0.1]])
    assert summary_metrics(y, svm_like)["accuracy"] == 1.0


def test_summary_metrics_rejects_bad_shapes():
    with pytest.raises(ValueError):
        summary_metrics(np.array([0, 1]), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        summary_metrics(np.array([0, 1]), np.zeros((2, 3)), num_classes=4)


def test_per_class_metrics_reports_support_and_none_for_empty():
    rows = per_class_metrics(np.array([0, 0, 1]), np.array([0, 1, 1]), ["a", "b", "c"])
    assert [r["support"] for r in rows] == [2, 1, 0]
    assert rows[0]["precision"] == 1.0 and rows[0]["recall"] == 0.5
    assert rows[2]["f1"] is None


def test_results_upsert_replaces_same_run_and_split(tmp_path):
    path = tmp_path / "reports" / "results.csv"
    m = {"n": 10, "classes": 2, "accuracy": 0.5, "macro_f1": 0.4, "weighted_f1": 0.5, "top3": 1}
    upsert_rows([make_row("a", "logreg", "val", m, {"C": 1}, commit="x")], path)
    upsert_rows([make_row("b", "cnn", "val", m, commit="x")], path)
    m2 = dict(m, macro_f1=0.9)
    table = upsert_rows([make_row("a", "logreg", "val", m2, {"C": 10}, commit="x")], path)

    assert list(table.columns) == COLUMNS
    assert sorted(table["run_id"]) == ["a", "b"]
    a = pd.read_csv(path).set_index("run_id").loc["a"]
    assert a["macro_f1"] == pytest.approx(0.9)
    assert json.loads(a["hyperparams"]) == {"C": 10}


def test_resolve_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("PLANTDX_ARTIFACTS", str(tmp_path))
    assert resolve_artifact("reports/results.csv") == tmp_path / "reports" / "results.csv"
    assert resolve_artifact(tmp_path / "x.csv") == tmp_path / "x.csv"


def test_export_for_repo_uses_repository_paths(tmp_path, monkeypatch):
    import zipfile

    from plantdx.results import export_for_repo, results_path

    monkeypatch.setenv("PLANTDX_ARTIFACTS", str(tmp_path / "art"))
    m = {"n": 1, "classes": 1, "accuracy": 1, "macro_f1": 1, "weighted_f1": 1, "top3": 1}
    upsert_rows([make_row("r1", "cnn", "val", m, commit="x")], results_path())
    run = tmp_path / "art" / "runs" / "r1"
    run.mkdir(parents=True)
    (run / "metrics_val.json").write_text("{}")
    (run / "best.keras").write_text("big")  # models are not exported
    fig = tmp_path / "curves_r1.png"
    fig.write_bytes(b"png")

    out = export_for_repo(["r1", "missing"], tmp_path / "out.zip", figures=[fig])
    with zipfile.ZipFile(out) as zf:
        assert sorted(zf.namelist()) == [
            "reports/figures/curves_r1.png",
            "reports/results.csv",
            "reports/runs/r1/metrics_val.json",
        ]


def test_plot_learning_curves(tmp_path):
    from plantdx.viz import plot_learning_curves

    hist = pd.DataFrame(
        {
            "epoch": [0, 1, 2],
            "loss": [2.0, 1.0, 0.5],
            "val_loss": [2.1, 1.2, 0.9],
            "accuracy": [0.3, 0.6, 0.8],
            "val_accuracy": [0.3, 0.5, 0.7],
            "macro_f1": [0.2, 0.5, 0.8],
            "val_macro_f1": [0.2, 0.6, 0.55],
        }
    )
    hist.to_csv(tmp_path / "history.csv", index=False)
    out = plot_learning_curves(tmp_path / "history.csv", tmp_path / "c.png", title="t")
    assert out.is_file() and out.stat().st_size > 1000


def test_select_final_by_val_macro_f1_ties_to_smaller_model():
    from plantdx.results import select_final

    m = {"n": 1, "classes": 1, "accuracy": 0.9, "weighted_f1": 0.9, "top3": 1.0}
    table = pd.DataFrame(
        [
            make_row("head", "m", "val", dict(m, macro_f1=0.95), commit="x", size_mb=9.8),
            make_row("ft", "m", "val", dict(m, macro_f1=0.98), commit="x", size_mb=9.8),
            make_row("ft_small", "m", "val", dict(m, macro_f1=0.98), commit="x", size_mb=5.0),
            make_row("other", "m", "val", dict(m, macro_f1=0.99), commit="x", size_mb=1.0),
            make_row("ft", "m", "test", dict(m, macro_f1=1.0), commit="x", size_mb=9.8),
        ]
    )
    best, ranked = select_final(table, ["head", "ft", "ft_small"])
    assert best == "ft_small"  # "other" is not a candidate; test rows are ignored
    assert ranked["run_id"].tolist() == ["ft_small", "ft", "head"]
    with pytest.raises(ValueError):
        select_final(table, ["missing"])


def test_update_run_fields(tmp_path):
    from plantdx.results import update_run_fields

    path = tmp_path / "r.csv"
    m = {"n": 1, "classes": 1, "accuracy": 1, "macro_f1": 1, "weighted_f1": 1, "top3": 1}
    upsert_rows(
        [
            make_row("a", "cnn", "val", m, commit="x", size_mb=23.8),
            make_row("b", "cnn", "val", m, commit="x", size_mb=1.0),
        ],
        path,
    )
    table = update_run_fields(path, "a", size_mb=4.86, cpu_ms=12.5)
    a = table.set_index("run_id").loc["a"]
    assert a["size_mb"] == 4.86 and a["cpu_ms"] == 12.5
    assert table.set_index("run_id").loc["b", "size_mb"] == 1.0
    with pytest.raises(KeyError):
        update_run_fields(path, "a", nonsense=1)


def test_family_winners_and_final_across_families():
    from plantdx.evaluate import family_winners
    from plantdx.results import select_final

    m = {"n": 1, "classes": 1, "accuracy": 0.9, "weighted_f1": 0.9, "top3": 1.0}
    table = pd.DataFrame(
        [
            make_row("cnn", "cnn_scratch", "val", dict(m, macro_f1=0.973), commit="x", size_mb=4.9),
            make_row(
                "mb_a", "mobilenetv2", "val", dict(m, macro_f1=0.955), commit="x", size_mb=9.8
            ),
            make_row(
                "mb_b", "mobilenetv2", "val", dict(m, macro_f1=0.981), commit="x", size_mb=9.8
            ),
            make_row("lr", "logreg", "val", dict(m, macro_f1=0.874), commit="x", size_mb=2.0),
        ]
    )
    candidates = ["cnn", "mb_a", "mb_b"]
    assert family_winners(table, candidates) == {"cnn_scratch": "cnn", "mobilenetv2": "mb_b"}
    assert select_final(table, candidates)[0] == "mb_b"
    # the CNN wins when it is better: the final model is chosen across families
    table.loc[table.run_id == "mb_b", "macro_f1"] = 0.96
    assert select_final(table, candidates)[0] == "cnn"
