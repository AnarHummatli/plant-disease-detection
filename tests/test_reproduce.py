"""The reproduction check (plan step 7, D-024): comparison logic, lock file and report."""

import json
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest

from plantdx import reproduce as R

ROOT = Path(__file__).resolve().parents[1]


def test_examples_cover_the_demo_folder():
    names = sorted(p.name for p in (ROOT / "app" / "gradio" / "examples").glob("*.jpg"))
    assert names == sorted(R.EXAMPLES)
    assert R.EXAMPLE_NOT_SURE in R.EXAMPLES


def test_reference_files_exist():
    """The numbers the check compares against are committed."""
    for path in (
        ROOT / "reports" / "runs" / R.FINAL_RUN / "metrics_test.json",
        ROOT / "reports" / "runs" / R.FINAL_RUN / "env.json",
        ROOT / "reports" / "runs" / R.TRAIN_REFERENCE / "metrics_val.json",
        ROOT / "reports" / "analysis" / "confusion_test.csv",
        ROOT / "reports" / "analysis.json",
        ROOT / R.TRAIN_CONFIG,
        *(ROOT / "splits" / name for name in R.SPLIT_FILES),
    ):
        assert path.is_file(), path


def test_dependency_closure_follows_requirements():
    found = R.dependency_closure({"pandas": set()})
    assert found["pandas"] == metadata.version("pandas")
    assert "numpy" in found and "python-dateutil" in found
    assert R.dependency_closure({"not-a-real-package": set()}) == {}


def test_write_lock(tmp_path):
    try:
        metadata.metadata("plantdx")
    except metadata.PackageNotFoundError:
        pytest.skip("plantdx is not installed (pip install -e .)")
    n = R.write_lock(tmp_path / "requirements.lock")
    lines = (tmp_path / "requirements.lock").read_text().splitlines()
    pins = [line for line in lines if not line.startswith("#")]
    assert n == len(pins) > 5
    assert all(line.count("==") == 1 for line in pins)
    order = [line.split("==")[0] for line in pins]
    assert order == sorted(order)
    names = set(order)
    assert {"numpy", "pandas", "scikit-learn", "pyyaml"} <= names and "plantdx" not in names


def test_confusion_and_changed_predictions():
    y = np.array([0, 0, 1, 2])
    a = R.confusion(y, np.array([0, 0, 1, 2]), 3)
    b = R.confusion(y, np.array([0, 1, 1, 2]), 3)
    assert a.sum() == 4 and np.trace(a) == 4
    assert R.changed_predictions(a, a) == 0
    assert R.changed_predictions(a, b) == 1


def test_not_sure_numbers():
    probs = np.array([[0.99, 0.01], [0.6, 0.4], [0.3, 0.7], [0.98, 0.02]])
    y = np.array([0, 1, 1, 1])  # the second and the last are wrong
    ns = R.not_sure_numbers(y, probs, 0.947)
    assert ns["coverage"] == 0.5  # 0.99 and 0.98 answered
    assert ns["accuracy_answered"] == 0.5
    assert ns["errors_caught"] == 1


def test_run_writes_report_and_keeps_going_after_a_crash(tmp_path, monkeypatch):
    def boom():
        raise RuntimeError("no network")

    monkeypatch.setattr(R, "check_environment", lambda out: [R.check("env", "INFO", 1, 1)])
    monkeypatch.setattr(R, "check_splits", boom)
    monkeypatch.setattr(R, "check_release_model", lambda root: [R.check("m", "PASS", 0.98, 0.98)])
    monkeypatch.setattr(R, "check_examples", lambda: R.check("demo", "PASS"))
    monkeypatch.setattr(R, "check_browser_export", lambda: R.check("web", "SKIP"))
    rows = R.run("data", tmp_path)

    assert [r["status"] for r in rows] == ["INFO", "FAIL", "PASS", "PASS", "SKIP"]
    assert "no network" in rows[1]["note"]
    md = (tmp_path / "reproduction.md").read_text()
    assert "2 checks passed, 1 failed" in md and "| m | 0.9800 | 0.9800 | PASS |" in md
    saved = json.loads((tmp_path / "reproduction.json").read_text())
    assert len(saved["checks"]) == 5 and saved["info"]["Training run included"].startswith("no")


def test_main_exits_with_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "run", lambda *a, **k: [R.check("x", "FAIL")])
    with pytest.raises(SystemExit) as exc:
        R.main(["--data-root", "d", "--out", str(tmp_path)])
    assert exc.value.code == 1
