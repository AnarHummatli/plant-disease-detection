"""The shared results table: one row per (run, split) (plan steps 3.x-5.2).

Every model, classical or neural, appends its numbers here with the same columns, so the
comparison table in the report is generated from one file. A re-run with the same run id
replaces its old rows instead of adding duplicates.

The file lives under the artifacts folder (Google Drive on Colab), because the Colab copy of the
repository is reset on every sync (D-009). Notebooks download it (export_for_repo) so it can be
committed to reports/results.csv.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from plantdx.paths import artifacts_dir, repo_root

COLUMNS = [
    "run_id",
    "model",
    "split",
    "n",
    "classes",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "top3",
    "params",
    "size_mb",
    "train_seconds",
    "epochs",
    "hyperparams",
    "commit",
    "date",
]


def resolve_artifact(path: str | Path) -> Path:
    """Absolute paths stay as they are; relative ones resolve against artifacts_dir()."""
    path = Path(path)
    return path if path.is_absolute() else artifacts_dir() / path


def results_path(cfg: Mapping[str, Any] | None = None) -> Path:
    rel = (cfg or {}).get("paths", {}).get("results_csv", "reports/results.csv")
    return resolve_artifact(rel)


def git_commit() -> str:
    """Short hash of the checked-out commit, or '' outside a git repository."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root()), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def make_row(
    run_id: str,
    model: str,
    split: str,
    metrics: Mapping[str, Any],
    hyperparams: Mapping[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    row = {c: None for c in COLUMNS}
    row.update({k: v for k, v in metrics.items() if k in COLUMNS})
    row.update({k: v for k, v in extra.items() if k in COLUMNS})
    row.update(
        run_id=run_id,
        model=model,
        split=split,
        hyperparams=json.dumps(dict(hyperparams or {}), sort_keys=True),
        commit=extra.get("commit") or git_commit(),
        date=extra.get("date") or dt.date.today().isoformat(),
    )
    return row


def upsert_rows(rows: list[Mapping[str, Any]], path: str | Path) -> pd.DataFrame:
    """Write `rows` into the CSV at `path`, replacing earlier rows with the same (run_id, split)."""
    path = Path(path)
    new = pd.DataFrame(list(rows), columns=COLUMNS)
    if path.exists():
        old = pd.read_csv(path)
        keys = set(zip(new["run_id"], new["split"], strict=True))
        keep = [(r, s) not in keys for r, s in zip(old["run_id"], old["split"], strict=True)]
        table = pd.concat([old[keep], new], ignore_index=True)
    else:
        table = new
    table = table.reindex(columns=COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


# Small per-run files worth committing (models and score arrays stay on Drive, D-005).
EXPORT_FILES = ("metrics_val.json", "history.csv", "grid.csv", "config.yaml", "env.json")


def export_for_repo(
    run_ids: list[str],
    dest_zip: str | Path,
    cfg: Mapping[str, Any] | None = None,
    figures: list[str | Path] | None = None,
) -> Path:
    """Zip results.csv, the small files of `run_ids` and `figures` with repository paths.

    Unzipping the archive at the repository root puts them under reports/, ready to commit:
    reports/results.csv, reports/runs/<run_id>/..., reports/figures/<name>.
    """
    import zipfile

    from plantdx.paths import run_dir

    dest_zip = Path(dest_zip)
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        table = results_path(cfg)
        if table.exists():
            zf.write(table, "reports/results.csv")
        for run_id in run_ids:
            folder = run_dir(run_id, create=False)
            for name in EXPORT_FILES:
                if (folder / name).is_file():
                    zf.write(folder / name, f"reports/runs/{run_id}/{name}")
        for fig in figures or []:
            fig = Path(fig)
            zf.write(fig, f"reports/figures/{fig.name}")
    return dest_zip
