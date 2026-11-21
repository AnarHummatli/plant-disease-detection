# Roadmap

The project was planned in eight stages before any code was written (plan v1.0, 18 November
2026). References such as "plan step 2.5", "plan 4.2" or "step 7.2" in the code, notebooks and
`docs/decisions.md` point to the steps below. Where the work departed from the plan, the
decision log says why. **Status: all stages done.**

| Stage | Step | What | Where |
|---|---|---|---|
| **1 · Foundation** | 1.1 | Repository, package layout, configs | `src/plantdx/`, `configs/`, `pyproject.toml` |
| | 1.2 | Environment report and Colab bootstrap | `plantdx.env`, notebook 00 |
| | 1.3 | Tests, linting, continuous integration | `tests/`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml` |
| | 1.4 | Problem, users, input/output, success criteria | [`problem_statement.md`](problem_statement.md) |
| **2 · Data** | 2.1 | Download the dataset, check sizes and licence | `plantdx.data.images`, `plantdx.data.source`, D-001 |
| | 2.2 | Audit and exploratory analysis | notebook 01 |
| | 2.3 | Leaf ids, near-duplicates, same-leaf pairs | `plantdx.data.duplicates`, `plantdx.data.sameleaf`, notebooks 02 and 04, D-004, D-006, D-010 |
| | 2.4 | Leaf-grouped train / val / test split and leakage tests | `plantdx.data.split`, `plantdx.data.prepare`, `splits/`, `tests/test_splits.py` |
| | 2.5 | Input pipeline and augmentation | `plantdx.data.pipeline`, notebook 03, D-008 |
| | 2.6 | Data card | [`data_card.md`](data_card.md), `plantdx.data.card` |
| **3 · Baselines and CNN** | 3.1 | Hand-crafted features (colour, HOG, LBP) | `plantdx.baselines.features`, D-011 |
| | 3.2 | Logistic regression and linear SVM | `plantdx.baselines`, notebook 05 |
| | 3.3 | CNN trained from scratch | `plantdx.models.cnn`, D-012 |
| | 3.3a | One trainer and metrics for every Keras model | `plantdx.train`, `plantdx.metrics`, D-014 |
| **4 · Transfer learning and test** | 4.1 | MobileNetV2 with a frozen backbone | `plantdx.models.mobilenet`, D-016 |
| | 4.2 | Fine-tuning | `configs/mobilenetv2_ft*.yaml`, notebook 06, D-017 |
| | 4.3 | Selection on validation | `plantdx.evaluate`, D-015 |
| | 4.4 | One test evaluation per chosen model; release of the final model | notebook 07, release v1.0 |
| **5 · Evaluation and error analysis** | 5.1 | Metrics, per-class table, confusion matrix, learning curves | `plantdx.analysis`, `reports/` |
| | 5.2 | Comparison of all models | [`reports/results.md`](../reports/results.md) |
| | 5.3 | Confusions, weakest classes, leakage check on the verified subset | [`error_analysis.md`](error_analysis.md) |
| | 5.4 | Failure cases | `reports/figures/failures.jpg` |
| | 5.5 | Grad-CAM (measured on leaf masks) and the "not sure" threshold | `plantdx.gradcam`, notebook 08, D-013, D-019, D-020, D-022 |
| **6 · Demo** | 6.1 | Upload app: verdict, top-3, Grad-CAM, "not sure" | `plantdx.predict`, `app/gradio/`, D-021 |
| | 6.2 | Run it locally and from Colab | `make demo`, notebook 09, release v1.1 |
| | 6.3 | Video for presenting without internet | release v1.2 |
| | 6.4 | Permanent public link | `app/web/`, Hugging Face Space and GitHub Pages, D-023 |
| **7 · Reproducibility** | 7.1 | README: results, one command per step | [`README.md`](../README.md) |
| | 7.2 | Exact package versions | `requirements.lock` |
| | 7.3 | Tests for every module, CI green | `tests/`, CI on Python 3.11–3.13 |
| | 7.4 | Clean-session reproduction check | `plantdx.reproduce`, notebook 10, [`reports/reproduction/`](../reports/reproduction/reproduction.md), D-024 |
| **8 · Write-up** | 8.1 | Technical report | [`report.pdf`](report.pdf), source [`report.md`](report.md) |
