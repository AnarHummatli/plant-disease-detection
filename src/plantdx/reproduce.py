"""Reproduction check: does a fresh environment give the reported results? (plan step 7, D-024)

    python -m plantdx.reproduce --data-root /content/data            # ~5 min on a T4
    python -m plantdx.reproduce --data-root /content/data --train    # + one training run, ~15 min

Every check compares a number from the repository (reports/, splits/) with the same number
computed again in the current environment, and passes only within a stated tolerance:

1. **Environment** (information): the installed versions against those the final model was
   trained with (reports/runs/<final>/env.json). Writes requirements.lock: the exact versions of
   every package that plantdx (all extras) and TensorFlow need, directly or indirectly.
2. **Splits**: rebuilt from the official index files and the committed grouping pairs into a
   temporary folder; every file must be byte-identical to splits/.
3. **Released model on test**: inference.keras from the release (SHA-256 checked) scored on the
   test split must give the reported macro-F1 and accuracy (full and leaf-verified part), the same
   confusion matrix and the same "not sure" numbers at 0.947.
   This re-computes published numbers of a frozen model; nothing is selected, tuned or changed
   from it, so the test-once rule (D-015) is kept (D-024).
4. **Demo**: the seven example photos get their expected answers (one of them "not sure").
5. **Browser export** (if tf2onnx is installed): the ONNX model gives the Keras answers.
6. **Training** (`--train`): configs/mobilenetv2_head.yaml trained again from scratch must reach
   the reported validation macro-F1 within 0.01 (GPU training is not bit-deterministic, D-018).

Writes reproduction.md, reproduction.json and requirements.lock to --out (default
reports/reproduction/) and exits with status 1 if any check fails.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plantdx.paths import repo_root

FINAL_RUN = "mobilenetv2_ft_all"
TRAIN_CONFIG = "configs/mobilenetv2_head.yaml"
TRAIN_REFERENCE = "mobilenetv2_head"
TRAIN_RUN = "repro_mobilenetv2_head"

# tolerances (D-024)
METRIC_TOL = 1e-3  # test metrics of the frozen model
CHANGED_TOL = 5  # test predictions allowed to differ (float differences between devices)
TRAIN_TOL = 0.01  # val macro-F1 of a model trained again (D-018)

SPLIT_FILES = (
    "train.csv",
    "val.csv",
    "test.csv",
    "classes.json",
    "dropped_train_overlap.csv",
    "summary.json",
)
PAIR_FILES = (
    "duplicate_pairs.csv",
    "sameleaf_pairs.csv",
    "duplicates_meta.json",
    "sameleaf_meta.json",
)
VERSION_KEYS = ("python", "tensorflow", "keras", "numpy", "pandas", "scikit-learn")

# the demo examples (app/gradio/examples) and the class the released model gives each
EXAMPLES = {
    "apple_cedar_apple_rust.jpg": "Apple___Cedar_apple_rust",
    "grape_black_rot.jpg": "Grape___Black_rot",
    "pepper_bell_healthy.jpg": "Pepper,_bell___healthy",
    "potato_early_blight.jpg": "Potato___Early_blight",
    "strawberry_leaf_scorch.jpg": "Strawberry___Leaf_scorch",
    "tomato_early_blight_part_of_leaf.jpg": "Tomato___Early_blight",
    "tomato_yellow_leaf_curl_virus.jpg": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}
EXAMPLE_NOT_SURE = "tomato_early_blight_part_of_leaf.jpg"


def check(name: str, status: str, reported: Any = None, reproduced: Any = None, note: str = ""):
    """One line of the report. status: PASS, FAIL, INFO or SKIP."""
    return {
        "check": name,
        "status": status,
        "reported": reported,
        "reproduced": reproduced,
        "note": note,
    }


def passed(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def fmt(x: Any) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def reports_dir() -> Path:
    return repo_root() / "reports"


# ------------------------------------------------------------------------------ environment
def _name(dist_name: str) -> str:
    return re.sub(r"[-_.]+", "-", dist_name).lower()


def dependency_closure(roots: dict[str, set[str]]) -> dict[str, str]:
    """{name: version} of the installed distributions that `roots` need, directly or not.

    `roots` maps a distribution name to the extras to include ({"plantdx": {"dev", ...}}).
    Requirements whose environment markers do not hold, or that are not installed, are skipped.
    """
    from packaging.requirements import Requirement

    found: dict[str, str] = {}
    todo = [(r, set(extras)) for r, extras in roots.items()]
    seen: set[tuple[str, frozenset[str]]] = set()
    while todo:
        name, extras = todo.pop()
        key = (_name(name), frozenset(extras))
        if key in seen:
            continue
        seen.add(key)
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        found[_name(dist.metadata["Name"])] = dist.version
        for spec in dist.requires or []:
            req = Requirement(spec)
            envs = [{"extra": e} for e in extras] or [{"extra": ""}]
            if req.marker and not any(req.marker.evaluate(env) for env in envs):
                continue
            todo.append((req.name, set(req.extras)))
    return found


def write_lock(path: Path) -> int:
    """requirements.lock for this environment; returns the number of packages."""
    plantdx_extras = set(metadata.metadata("plantdx").get_all("Provides-Extra") or [])
    roots = {"plantdx": plantdx_extras, "tensorflow": set(), "tensorflow-cpu": set()}
    versions = dependency_closure(roots)
    versions.pop("plantdx", None)
    lines = [
        "# Exact versions of the reproduction run (plan step 7.2), written by",
        "# `python -m plantdx.reproduce`: every package that plantdx (all extras) and TensorFlow",
        "# need, directly or indirectly. Reproduce with: pip install -r requirements.lock",
        f"# Python {platform.python_version()} on {platform.platform()}",
        *(f"{name}=={versions[name]}" for name in sorted(versions)),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(versions)


def gpu_description() -> str:
    try:
        import tensorflow as tf
    except ImportError:
        return "none (TensorFlow not installed)"
    names = [
        tf.config.experimental.get_device_details(d).get("device_name", d.name)
        for d in tf.config.list_physical_devices("GPU")
    ]
    return ", ".join(names) or "none"


def check_environment(out: Path) -> list[dict]:
    from plantdx.env import report

    now = report()
    trained = json.loads((reports_dir() / "runs" / FINAL_RUN / "env.json").read_text())
    rows = []
    for key in VERSION_KEYS:
        a, b = trained.get(key), now.get(key)
        same = a is not None and b is not None and str(a) == str(b)
        rows.append(check(f"version: {key}", "INFO", a, b, "same" if same else "differs"))
    rows.append(check("GPU", "INFO", "Tesla T4 (training)", gpu_description()))
    n = write_lock(out / "requirements.lock")
    rows.append(check("requirements.lock", "INFO", None, f"{n} packages"))
    return rows


# ----------------------------------------------------------------------------------- splits
def check_splits() -> dict:
    from plantdx.data import prepare

    committed = repo_root() / "splits"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in PAIR_FILES:
            if (committed / name).is_file():
                shutil.copyfile(committed / name, tmp / name)
        with contextlib.redirect_stdout(io.StringIO()):
            prepare.main(["--out", str(tmp)])
        differ = [n for n in SPLIT_FILES if (tmp / n).read_bytes() != (committed / n).read_bytes()]
    return check(
        "splits rebuilt from the official index files",
        passed(not differ),
        f"{len(SPLIT_FILES)} committed files",
        "identical" if not differ else f"differ: {', '.join(differ)}",
    )


# ------------------------------------------------------------------------- released model
def confusion(y: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros((k, k), dtype=np.int64)
    np.add.at(m, (y, pred), 1)
    return m


def changed_predictions(a: np.ndarray, b: np.ndarray) -> int:
    """Lower bound of the predictions that differ between two confusion matrices."""
    return int(np.abs(a - b).sum() // 2)


def not_sure_numbers(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    conf = probs.max(axis=1)
    answered = conf >= threshold
    right = probs.argmax(axis=1) == y
    return {
        "coverage": float(answered.mean()),
        "accuracy_answered": float(right[answered].mean()),
        "errors_caught": int((~right & ~answered).sum()),
    }


def score_release_model(data_root: str | Path) -> dict[str, Any]:
    import keras

    from plantdx.data.images import missing_files
    from plantdx.data.pipeline import load_split, make_dataset
    from plantdx.predict import model_path

    frame = load_split("test")
    missing = missing_files(data_root, "color", frame["path"].head(200))
    if missing:
        raise FileNotFoundError(f"test images not found under {data_root}: {missing[:3]} ...")
    path = model_path()  # downloads release v1.0 once, SHA-256 checked
    model = keras.models.load_model(path)
    ds = make_dataset(frame, data_root, batch_size=64)
    t0 = time.perf_counter()
    probs = np.asarray(model.predict(ds, verbose=0), dtype=np.float32)
    return {
        "probs": probs,
        "y": frame["label_index"].to_numpy(),
        "verified": (frame["leaf_source"] == "map").to_numpy(),
        "seconds": time.perf_counter() - t0,
    }


def check_release_model(data_root: str | Path) -> list[dict]:
    from plantdx.data.source import CLASSES
    from plantdx.metrics import summary_metrics
    from plantdx.predict import NOT_SURE_THRESHOLD

    s = score_release_model(data_root)
    probs, y, verified = s["probs"], s["y"], s["verified"]
    reported = json.loads((reports_dir() / "runs" / FINAL_RUN / "metrics_test.json").read_text())
    rows = []
    for part, mask in (("test", slice(None)), ("test_verified", verified)):
        now = summary_metrics(y[mask], probs[mask])
        for metric in ("macro_f1", "accuracy"):
            a, b = reported[part][metric], now[metric]
            rows.append(
                check(f"released model, {part}: {metric}", passed(abs(a - b) <= METRIC_TOL), a, b)
            )
    old = pd.read_csv(reports_dir() / "analysis" / "confusion_test.csv", index_col=0)
    old = old.loc[list(CLASSES), list(CLASSES)].to_numpy()
    new = confusion(y, probs.argmax(1), len(CLASSES))
    changed = changed_predictions(old, new)
    rows.append(
        check(
            "released model, test: confusion matrix",
            passed(changed <= CHANGED_TOL),
            f"{int(old.sum()) - int(np.trace(old))} errors",
            f"{int(new.sum()) - int(np.trace(new))} errors, {changed} predictions differ",
            f"tolerance {CHANGED_TOL} of {len(y):,}",
        )
    )
    ns_rep = json.loads((reports_dir() / "analysis.json").read_text())["not_sure"]["test"]
    ns_now = not_sure_numbers(y, probs, NOT_SURE_THRESHOLD)
    for key in ("coverage", "accuracy_answered"):
        a, b = ns_rep[key], ns_now[key]
        rows.append(
            check(
                f'"not sure" at {NOT_SURE_THRESHOLD}: {key}', passed(abs(a - b) <= METRIC_TOL), a, b
            )
        )
    rows.append(
        check(
            "released model, test: scoring time",
            "INFO",
            None,
            f"{s['seconds']:.0f} s for {len(y):,} images",
        )
    )
    return rows


# ------------------------------------------------------------------------------------ demo
def check_examples() -> dict:
    from plantdx.predict import Predictor, model_path

    folder = repo_root() / "app" / "gradio" / "examples"
    predictor = Predictor(model_path())
    wrong = []
    for name, label in EXAMPLES.items():
        pred = predictor.predict(folder / name, explain=False)
        if pred.label != label or pred.not_sure != (name == EXAMPLE_NOT_SURE):
            wrong.append(f"{name}: {pred.label} {pred.confidence:.3f}")
    return check(
        "demo: the seven example photos",
        passed(not wrong),
        "7 expected answers (1 not sure)",
        "all as expected" if not wrong else "; ".join(wrong),
    )


def check_browser_export() -> dict:
    try:
        import tf2onnx  # noqa: F401
    except ImportError:
        return check("browser export (ONNX)", "SKIP", note="install the [web] extra")
    from plantdx.web_export import build_site

    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        c = build_site(tmp)  # raises if ONNX and Keras disagree
    return check(
        "browser export (ONNX)",
        "PASS",
        "same top class, probs < 1e-4",
        f"probs {c['probs']:.1e}, Grad-CAM {c['cam']:.1e}",
    )


# -------------------------------------------------------------------------------- training
def check_training(data_root: str | Path) -> dict:
    from plantdx.train import train

    reference = json.loads(
        (reports_dir() / "runs" / TRAIN_REFERENCE / "metrics_val.json").read_text()
    )
    a = reference["metrics"]["macro_f1"]
    report = train(repo_root() / TRAIN_CONFIG, data_root, run_id=TRAIN_RUN, verbose=2)
    b = report["metrics"]["macro_f1"]
    return check(
        f"training again: {TRAIN_REFERENCE}, val macro-F1",
        passed(abs(a - b) <= TRAIN_TOL),
        a,
        b,
        f"tolerance {TRAIN_TOL} (D-018); {report['train_seconds'] / 60:.1f} min, "
        f"best epoch {report['best_epoch']}/{report['epochs_trained']}",
    )


# ---------------------------------------------------------------------------------- report
def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root()), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def render(rows: list[dict], info: dict[str, str]) -> str:
    fails = sum(r["status"] == "FAIL" for r in rows)
    passes = sum(r["status"] == "PASS" for r in rows)
    lines = [
        "# Reproduction check",
        "",
        "Generated by `python -m plantdx.reproduce` (plan step 7, `docs/decisions.md` D-024).",
        "",
        *(f"- **{k}:** {v}" for k, v in info.items()),
        f"- **Result:** {passes} checks passed, {fails} failed",
        "",
        "| Check | Reported | Reproduced | Result | Note |",
        "|---|---|---|---|---|",
        *(
            f"| {r['check']} | {fmt(r['reported'])} | {fmt(r['reproduced'])} | {r['status']} "
            f"| {r['note']} |"
            for r in rows
        ),
        "",
    ]
    return "\n".join(lines)


def run(data_root: str | Path, out: str | Path, do_train: bool = False) -> list[dict]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    steps = [
        ("environment", lambda: check_environment(out)),
        ("splits", check_splits),
        ("released model on test", lambda: check_release_model(data_root)),
        ("demo examples", check_examples),
        ("browser export", check_browser_export),
    ]
    if do_train:
        steps.append(("training", lambda: check_training(data_root)))
    rows: list[dict] = []
    for title, fn in steps:
        print(f"== {title}", flush=True)
        try:
            result = fn()
        except Exception as err:  # a crashed check is a failed check, the others still run
            result = check(title, "FAIL", note=f"{type(err).__name__}: {err}")
        for r in result if isinstance(result, list) else [result]:
            line = f"{fmt(r['reported'])} -> {fmt(r['reproduced'])}"
            print(f"   {r['status']:4s}  {r['check']}: {line}", flush=True)
            rows.append(r)
    info = {
        "Date": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "Commit": git_commit(),
        "Python": f"{platform.python_version()} on {platform.platform()}",
        "Training run included": "yes" if do_train else "no (`--train` adds it)",
    }
    (out / "reproduction.md").write_text(render(rows, info), encoding="utf-8")
    (out / "reproduction.json").write_text(
        json.dumps({"info": info, "checks": rows}, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True, help="folder that contains raw/color/")
    parser.add_argument("--out", default=str(repo_root() / "reports" / "reproduction"))
    parser.add_argument("--train", action="store_true", help=f"also train {TRAIN_CONFIG} (~15 min)")
    args = parser.parse_args(argv)
    rows = run(args.data_root, args.out, args.train)
    fails = [r for r in rows if r["status"] == "FAIL"]
    print(f"\n{len(rows) - len(fails)} of {len(rows)} lines without failure; report: {args.out}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
