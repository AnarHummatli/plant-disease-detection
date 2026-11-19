"""Stage 5: error analysis, Grad-CAM and the "not sure" threshold in one run (plan 5.1-5.5).

    python -m plantdx.analyze --data-root /content/data

Inputs (all saved earlier, nothing is trained or selected on test here):
<artifacts>/runs/<final>/inference.keras, val_scores.npz (training), test_scores.npz and
metrics_test.json (the one-time test evaluation, D-015), reports/final_model.json, and the colour
and segmented images (the segmented variant only provides leaf masks).

Outputs, under <out>/ (default <artifacts>/stage5/, laid out like the repository):
reports/results.md (comparison and analysis tables), reports/analysis.json (key numbers),
reports/analysis/*.csv, reports/figures/{confusion_matrix.png, coverage_not_sure.png,
failures.jpg, gradcam_examples.jpg, gradcam_on_leaf.png, dup_leak_examples.jpg}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plantdx import analysis as A
from plantdx.data.pipeline import load_split
from plantdx.data.source import CLASSES
from plantdx.metrics import summary_metrics
from plantdx.paths import artifacts_dir, repo_root, run_dir
from plantdx.results import results_path

CAM_PER_CLASS = 15  # test images per class for the Grad-CAM-on-leaf check (D-020)
SEED = 42


def short(name: str) -> str:
    crop, _, disease = name.partition("___")
    return f"{crop.split('_(')[0].replace('_', ' ')} · {disease.replace('_', ' ')}"


# ------------------------------------------------------------------------------ images
def load_images(root, rel_paths: list[str], size: int = 224) -> np.ndarray:
    """uint8 [n, size, size, 3], decoded and resized exactly like the training pipeline."""
    from plantdx.data.images import image_path
    from plantdx.data.pipeline import _decode

    return np.stack([_decode(str(image_path(root, "color", p)), size).numpy() for p in rel_paths])


def segmented_index(root) -> dict[str, Path]:
    """'<class>/<file stem, lower case>' -> segmented image path (files end in _final_masked)."""
    index = {}
    base = Path(root) / "raw" / "segmented"
    for f in base.glob("*/*"):
        stem = f.stem.replace("_final_masked", "").lower()
        index[f"{f.parent.name}/{stem}"] = f
    return index


def leaf_masks(seg_paths: list[Path], size: int = 224) -> np.ndarray:
    from PIL import Image

    from plantdx.gradcam import leaf_mask

    out = []
    for p in seg_paths:
        with Image.open(p) as im:
            rgb = np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.NEAREST))
        out.append(leaf_mask(rgb))
    return np.stack(out)


# --------------------------------------------------------------------------- markdown
def md_table(df: pd.DataFrame, floats: str = "{:.4f}") -> str:
    def fmt(v):
        if isinstance(v, float):
            return "—" if np.isnan(v) else floats.format(v)
        return str(v)

    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


MODEL_NAMES = {
    "baseline_logreg": "Colour/HOG/LBP + logistic regression",
    "baseline_linsvm": "Colour/HOG/LBP + linear SVM",
    "cnn_scratch": "CNN from scratch",
    "mobilenetv2_head": "MobileNetV2, frozen backbone",
    "mobilenetv2_ft": "MobileNetV2, last 38 layers, 1e-5",
    "mobilenetv2_ft_lr3e-5": "MobileNetV2, last 38 layers, 3e-5",
    "mobilenetv2_ft_lr1e-4": "MobileNetV2, last 38 layers, 1e-4",
    "mobilenetv2_ft_all": "MobileNetV2, whole backbone, 3e-5",
}


def comparison_table(table: pd.DataFrame, final: str) -> pd.DataFrame:
    val = table[table.split == "val"].set_index("run_id")
    test = table[table.split == "test"].set_index("run_id")
    ver = table[table.split == "test_verified"].set_index("run_id")
    rows = []
    for run_id in MODEL_NAMES:
        if run_id not in val.index:
            continue
        v = val.loc[run_id]
        t = test.loc[run_id] if run_id in test.index else None
        q = ver.loc[run_id] if run_id in ver.index else None
        params = v.get("params")
        rows.append(
            {
                "Model": MODEL_NAMES[run_id] + (" **(final)**" if run_id == final else ""),
                "Val macro-F1": float(v["macro_f1"]),
                "Val acc.": float(v["accuracy"]),
                "Test macro-F1": float(t["macro_f1"]) if t is not None else float("nan"),
                "Test acc.": float(t["accuracy"]) if t is not None else float("nan"),
                "Test weighted-F1": float(t["weighted_f1"]) if t is not None else float("nan"),
                "Test top-3": float(t["top3"]) if t is not None else float("nan"),
                "Verified macro-F1": float(q["macro_f1"]) if q is not None else float("nan"),
                "Params": "—" if pd.isna(params) else f"{int(params):,}",
                "MB": f"{float(v['size_mb']):.2f}",
                "CPU ms*": "—" if pd.isna(v.get("cpu_ms")) else f"{float(v['cpu_ms']):.0f}",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------- run
def run(
    data_root,
    out: str | Path | None = None,
    final: str | None = None,
    cam_per_class: int = CAM_PER_CLASS,
    splits_dir=None,
    duplicate_pairs: str | Path | None = None,
) -> dict[str, Any]:
    import keras

    from plantdx import gradcam as G
    from plantdx import viz

    out = Path(out) if out else artifacts_dir() / "stage5"
    figs = out / "reports" / "figures"
    tables = out / "reports" / "analysis"
    figs.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    reports = results_path().parent
    final = final or json.loads((reports / "final_model.json").read_text())["run_id"]
    folder = run_dir(final, create=False)
    classes = list(CLASSES)
    n_cls = len(classes)

    val_frame, test_frame = load_split("val", splits_dir), load_split("test", splits_dir)
    with np.load(folder / "val_scores.npz") as z:
        p_val, y_val = z["probs"].astype(np.float64), z["y"]
    with np.load(folder / "test_scores.npz") as z:
        p_test, y_test, verified = z["scores"].astype(np.float64), z["y"], z["verified"]
    assert np.array_equal(y_val, val_frame["label_index"].to_numpy()), "val order mismatch"
    assert np.array_equal(y_test, test_frame["label_index"].to_numpy()), "test order mismatch"
    metrics_test = json.loads((folder / "metrics_test.json").read_text())
    pred_test = p_test.argmax(1)

    # 5.1 confusion matrix --------------------------------------------------------------
    cm = A.confusion(y_test, pred_test, n_cls)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(tables / "confusion_test.csv")
    viz.plot_confusion(
        A.normalize_rows(cm),
        classes,
        figs / "confusion_matrix.png",
        title=f"{final}: test split, row-normalised (off-diagonal shares >= 0.005 printed)",
    )

    # 5.3 confusions, weakest classes, full vs verified -------------------------------
    conf = A.top_confusions(cm, classes, k=5)
    conf.to_csv(tables / "top_confusions.csv", index=False)
    weak = A.weakest_classes(metrics_test["per_class"], k=5)
    weak.to_csv(tables / "weakest_classes.csv", index=False)
    pd.DataFrame(metrics_test["per_class"]).to_csv(tables / "per_class_test.csv", index=False)

    verified_classes = np.unique(y_test[verified])
    same_classes = np.isin(y_test, verified_classes)

    def part(mask):
        return summary_metrics(y_test[mask], p_test[mask]) if mask.any() else None

    gap = {
        "full_38": part(np.ones(len(y_test), bool)),
        "full_same_30_classes": part(same_classes),
        "verified_30": part(verified),
        "unverified_in_30_classes": part(same_classes & ~verified),
        "missing_classes": [classes[i] for i in sorted(set(range(n_cls)) - set(verified_classes))],
    }

    # 5.5 "not sure" threshold, chosen on val ------------------------------------------
    threshold = A.choose_threshold(p_val, y_val, A.TARGET_ACCURACY)
    cur_val, cur_test = A.coverage_curve(p_val, y_val), A.coverage_curve(p_test, y_test)
    cur_val.to_csv(tables / "coverage_val.csv", index=False)
    cur_test.to_csv(tables / "coverage_test.csv", index=False)
    not_sure = {
        "rule": f"smallest threshold with >= {A.TARGET_ACCURACY:.1%} accuracy on answered "
        "val images",
        "val": A.apply_threshold(p_val, y_val, threshold),
        "test": A.apply_threshold(p_test, y_test, threshold),
    }
    viz.plot_coverage(
        {"val": cur_val, "test": cur_test},
        threshold,
        A.TARGET_ACCURACY,
        figs / "coverage_not_sure.png",
    )

    # Grad-CAM -------------------------------------------------------------------------
    model = keras.models.load_model(folder / "inference.keras")
    layer = G.default_layer(model)
    size = int(model.input_shape[1])  # images are fed at the model's own input size

    # 5.4 failure gallery: explain the predicted class
    fails = A.failure_cases(p_test, test_frame, conf, n=6)
    fails["predicted"] = [classes[i] for i in fails["predicted_index"]]
    fails["top3"] = [
        ", ".join(f"{short(classes[j])} {p_test[i, j]:.2f}" for j in np.argsort(-p_test[i])[:3])
        for i in fails["index"]
    ]
    fails.to_csv(tables / "failures.csv", index=False)
    if len(fails):
        imgs = load_images(data_root, fails["path"].tolist(), size)
        cams, _ = G.gradcam(model, imgs.astype(np.float32), layer)
    items = [
        {
            "image": imgs[k],
            "overlay": G.overlay(imgs[k], cams[k]),
            "caption": f"true: {short(r.label)}\npred: {short(r.predicted)} ({r.confidence:.2f})",
            "bad": True,
        }
        for k, r in enumerate(fails.itertuples())
    ]
    if items:
        viz.plot_cam_grid(
            items,
            figs / "failures.jpg",
            title="Misclassified test images and Grad-CAM (predicted class)",
        )

    # correct, confident examples from 9 different classes
    rng = np.random.default_rng(SEED)
    show_classes = np.linspace(0, n_cls - 1, 9).round().astype(int)
    picks = []
    for c in show_classes:
        idx = np.flatnonzero((y_test == c) & (pred_test == c) & (p_test.max(1) >= 0.9))
        if len(idx):
            picks.append(int(rng.choice(idx)))
    ex_items = []
    if picks:
        ex_imgs = load_images(data_root, test_frame["path"].iloc[picks].tolist(), size)
        ex_cams, _ = G.gradcam(model, ex_imgs.astype(np.float32), layer)
        ex_items = [
            {
                "image": ex_imgs[k],
                "overlay": G.overlay(ex_imgs[k], ex_cams[k]),
                "caption": f"{short(classes[y_test[i]])} ({p_test[i].max():.2f})",
                "bad": False,
            }
            for k, i in enumerate(picks)
        ]
        viz.plot_cam_grid(
            ex_items, figs / "gradcam_examples.jpg", title="Correct test predictions and Grad-CAM"
        )

    # D-020: how much of the heat map lies on the leaf (segmented variant = leaf mask)
    seg = segmented_index(data_root)
    sample = []
    for c in range(n_cls):
        idx = np.flatnonzero(y_test == c)
        sample.extend(rng.choice(idx, size=min(cam_per_class, len(idx)), replace=False).tolist())
    keys = [f"{p.split('/', 1)[0]}/{Path(p).stem.lower()}" for p in test_frame["path"].iloc[sample]]
    have = [(i, seg[k]) for i, k in zip(sample, keys, strict=True) if k in seg]
    leaf_rows = []
    for s in range(0, len(have), 64):
        chunk = have[s : s + 64]
        ids = [i for i, _ in chunk]
        im = load_images(data_root, test_frame["path"].iloc[ids].tolist(), size)
        masks = leaf_masks([p for _, p in chunk], size)
        cm_chunk, _ = G.gradcam(model, im.astype(np.float32), layer)
        for i, cam_i, m in zip(ids, cm_chunk, masks, strict=True):
            leaf_rows.append(
                {
                    "path": test_frame["path"].iloc[i],
                    "label": classes[y_test[i]],
                    "correct": bool(pred_test[i] == y_test[i]),
                    "leaf_area": float(m.mean()),
                    "cam_on_leaf": G.cam_on_mask(cam_i, m),
                }
            )
    leaf = pd.DataFrame(leaf_rows).dropna()
    leaf.to_csv(tables / "gradcam_on_leaf.csv", index=False)
    viz.plot_cam_on_leaf(leaf, figs / "gradcam_on_leaf.png")
    on_leaf = {
        "images": int(len(leaf)),
        "segmented_found": f"{len(have)} of {len(sample)}",
        "median_cam_on_leaf": float(leaf["cam_on_leaf"].median()),
        "mean_cam_on_leaf": float(leaf["cam_on_leaf"].mean()),
        "median_leaf_area": float(leaf["leaf_area"].median()),
        "share_cam_on_leaf_above_area": float((leaf["cam_on_leaf"] > leaf["leaf_area"]).mean()),
        "share_cam_on_leaf_below_half": float((leaf["cam_on_leaf"] < 0.5).mean()),
        "by_class_lowest": leaf.groupby("label")["cam_on_leaf"]
        .median()
        .sort_values()
        .head(5)
        .round(3)
        .to_dict(),
    }

    # D-006 figure: near-identical pairs that crossed train/test in the official split
    dup = pd.read_csv(duplicate_pairs or repo_root() / "splits" / "duplicate_pairs.csv")
    leaks = A.leak_pairs(dup, set(test_frame["path"]), k=8)
    n_leaks = len(A.leak_pairs(dup, set(test_frame["path"]), k=10_000))
    if len(leaks):
        viz.plot_pairs(
            leaks,
            data_root,
            "color",
            figs / "dup_leak_examples.jpg",
            labels=[f"r = {r:.3f}" for r in leaks["corr_hi"]],
            title="Near-identical re-shots of one leaf that crossed the official train/test "
            f"split: {len(leaks)} of {n_leaks} pairs, all removed from train (D-006)",
        )

    summary = {
        "final_model": final,
        "gradcam_layer": layer,
        "top_confusions": conf.drop(columns=["true_index", "predicted_index"]).to_dict("records"),
        "weakest_classes": weak[["class", "support", "precision", "recall", "f1"]].to_dict(
            "records"
        ),
        "full_vs_verified": gap,
        "not_sure": not_sure,
        "gradcam_on_leaf": on_leaf,
        "leak_pairs_crossing_official_split": n_leaks,
    }
    (out / "reports" / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_results_md(out / "reports" / "results.md", summary, final, conf, weak, metrics_test)
    print(json.dumps({k: summary[k] for k in ("not_sure", "gradcam_on_leaf")}, indent=2))
    return summary


def write_results_md(path: Path, s: dict, final: str, conf, weak, metrics_test) -> None:
    table = pd.read_csv(results_path())
    comp = comparison_table(table, final)
    g = s["full_vs_verified"]
    ns = s["not_sure"]
    ol = s["gradcam_on_leaf"]
    per_class = pd.DataFrame(metrics_test["per_class"])
    per_class_v = pd.DataFrame(metrics_test["per_class_verified"]).set_index("class")
    per_class["verified F1"] = pd.to_numeric(
        per_class["class"].map(per_class_v["f1"]), errors="coerce"
    )
    per_class = per_class[["class", "support", "precision", "recall", "f1", "verified F1"]]
    per_class["class"] = per_class["class"].map(short)

    def gap_row(name, m):
        return {
            "Test part": name,
            "Images": f"{m['n']:,}",
            "Classes": m["classes"],
            "Macro-F1": m["macro_f1"],
            "Accuracy": m["accuracy"],
        }

    parts = [
        ("Full test", "full_38"),
        ("Full test, the classes with verified leaves", "full_same_30_classes"),
        ("Leaf-verified images (leakage impossible)", "verified_30"),
        ("Unverified images of those classes", "unverified_in_30_classes"),
    ]
    gap_df = pd.DataFrame([gap_row(name, g[key]) for name, key in parts if g[key]])
    conf_df = conf[["true", "predicted", "count", "share_of_true"]].copy()
    conf_df["true"] = conf_df["true"].map(short)
    conf_df["predicted"] = conf_df["predicted"].map(short)
    weak_df = weak[["class", "support", "precision", "recall", "f1"]].copy()
    weak_df["class"] = weak_df["class"].map(short)
    t, v = ns["test"], ns["val"]
    lines = [
        "# Results",
        "",
        "Generated by `python -m plantdx.analyze` from `reports/results.csv` and the saved",
        f"predictions of the final model (`{final}`). Model selection used validation only; every",
        "test number comes from the single test evaluation (D-015).",
        "",
        "## 1. Model comparison",
        "",
        md_table(comp),
        "",
        "\\* CPU time per image at batch size 1, eager Keras on Colab's shared 2-core CPU,",
        "including feature extraction for the classical models. It overstates MobileNetV2's cost",
        "(D-015);",
        "the demo measures latency again. The other MobileNetV2 runs were selection candidates and",
        "were not scored on test.",
        "",
        "## 2. Full test vs leaf-verified test (residual leakage)",
        "",
        md_table(gap_df),
        "",
        f"Classes without verified leaves (only in the full test): "
        f"{', '.join(short(c) for c in g['missing_classes']) or 'none'}.",
        "",
        "Rows 2 and 3 cover the same classes, so their difference isolates leakage; row 1 vs 2 "
        "is the effect of the missing classes. In the last row compare **accuracy**: its macro-F1 "
        "averages only the classes that have unverified test images, several with few of them, "
        "and is not comparable.",
        "",
        "## 3. Most confused class pairs (test)",
        "",
        md_table(conf_df),
        "",
        "## 4. Weakest classes (test, F1)",
        "",
        md_table(weak_df),
        "",
        '## 5. "Not sure" answer',
        "",
        f"Rule fixed before looking at the result (D-019): {ns['rule']}. "
        f"Chosen threshold: **{ns['val']['threshold']:.4f}**.",
        "",
        "| Split | Answered | Accuracy (all) | Accuracy (answered) | Errors caught | Correct "
        "answers held back |",
        "|---|---|---|---|---|---|",
        f"| val | {v['coverage']:.1%} | {v['accuracy_all']:.4f} | {v['accuracy_answered']:.4f} | "
        f"{v['errors_caught']} of {v['errors_total']} | {v['correct_rejected']} |",
        f"| test | {t['coverage']:.1%} | {t['accuracy_all']:.4f} | {t['accuracy_answered']:.4f} | "
        f"{t['errors_caught']} of {t['errors_total']} | {t['correct_rejected']} |",
        "",
        "## 6. Does the model look at the leaf? (Grad-CAM, D-020)",
        "",
        f"On {ol['images']} test images (up to {CAM_PER_CLASS} per class), the share of the "
        "Grad-CAM mass that lies on the leaf (leaf mask from the dataset's segmented variant): "
        f"median **{ol['median_cam_on_leaf']:.2f}**, while the leaf covers a median "
        f"{ol['median_leaf_area']:.2f} of the image. For {ol['share_cam_on_leaf_above_area']:.0%} "
        f"of the images the heat map is more concentrated on the leaf than the leaf's area alone "
        f"would give; for {ol['share_cam_on_leaf_below_half']:.0%} less than half of it is on the "
        "leaf. Figures: `figures/gradcam_on_leaf.png`, `figures/gradcam_examples.jpg`, "
        "`figures/failures.jpg`.",
        "",
        "## 7. Per-class scores of the final model (test)",
        "",
        md_table(per_class, floats="{:.3f}"),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--final", default=None, help="default: reports/final_model.json")
    parser.add_argument("--cam-per-class", type=int, default=CAM_PER_CLASS)
    parser.add_argument("--splits-dir", default=None)
    args = parser.parse_args(argv)
    run(args.data_root, args.out, args.final, args.cam_per_class, args.splits_dir)


if __name__ == "__main__":
    main()
