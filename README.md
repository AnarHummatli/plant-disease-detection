# Plant Disease Detection · Yarpaq Həkimləri

Classifies a leaf photo into one of **38 plant disease / healthy classes** (14 crops) with a
CNN and MobileNetV2 transfer learning, evaluated on a **leaf-grouped, leakage-free split**, with
Grad-CAM explanations, a "not sure" answer for low-confidence inputs, and a Gradio demo.

Neural Networks group project, Azerbaijan State Oil and Industry University, Fall 2026 (topic 06).

> **Status:** stages 1–5 of 8 done (leakage-checked data and pipeline; baselines and a CNN from
> scratch; MobileNetV2 transfer learning scored on test once; error analysis with Grad-CAM and a
> "not sure" threshold). Next: the Gradio demo, a reproducibility check and the report.

## Final results (test split, evaluated once)

The final model is MobileNetV2 with the whole backbone fine-tuned (`mobilenetv2_ft_all`), chosen
on validation. Every model below was chosen on validation within its family and scored on the
test split exactly once (D-015). "Verified" is the part of the test split whose leaf ids are
known (8,033 images, 30 classes), where same-leaf leakage is impossible by construction.

| Model | Test macro-F1 | Test accuracy | Verified macro-F1 | Test top-3 | Size |
|---|---|---|---|---|---|
| **MobileNetV2, fine-tuned (final)** | **0.982** | **0.986** | **0.985** | **0.9996** | 9.8 MB |
| CNN from scratch | 0.967 | 0.977 | 0.971 | 0.9989 | 4.9 MB |
| Colour/HOG/LBP + logistic regression | 0.884 | 0.906 | 0.881 | 0.9822 | 2.0 MB |
| Colour/HOG/LBP + linear SVM | 0.873 | 0.901 | 0.869 | 0.9770 | 2.0 MB |

The neural models do not score lower on the verified part, so there is no sign that residual
leakage inflates the full-test numbers (the verified part lacks 8 classes, two of them among the
hardest, so the comparison is indicative, not a proof). Weakest test classes of the final model:
Tomato early blight (F1 0.885, recall 0.805), Corn Cercospora leaf spot (0.912), Tomato target
spot (0.935). Model file: `inference.keras` in the GitHub release v1.0.

## Error analysis

Full write-up: **[`docs/error_analysis.md`](docs/error_analysis.md)**; tables: `reports/results.md`.

- **Where it fails:** similar diseases of one crop (tomato early blight vs target spot and late
  blight; the two corn leaf-spot diseases) and, across crops, similar healthy leaves and
  tomato vs potato late blight (same pathogen).
- **What it looks at:** on 555 test images a median 69% of the Grad-CAM heat map lies on the leaf,
  which covers 47% of the image (leaf masks from the segmented variant) — the leaf, not the
  background.
- **Leakage check:** on the same 30 classes, leaf-verified test images score as high as all test
  images (macro-F1 0.9849 vs 0.9848): no sign of inflated results.
- **"Not sure":** below a confidence of 0.947 (chosen on validation) the model abstains; on test
  it then answers 96.8% of the images with 99.5% accuracy and catches two thirds of its errors.
  The errors it still makes are confident ones (calibration is future work).

![Misclassified test images with Grad-CAM](reports/figures/failures.jpg)

## Validation results (model selection)

All choices were made on the validation split only.

| Model | Val macro-F1 | Val accuracy | Top-3 | Training |
|---|---|---|---|---|
| Colour/HOG/LBP + logistic regression (C = 0.1) | 0.874 | 0.900 | 0.982 | 46 s (CPU) |
| Colour/HOG/LBP + linear SVM (C = 0.1) | 0.873 | 0.904 | 0.975 | 116 s (CPU) |
| CNN from scratch (1.18 M parameters) | 0.973 | 0.980 | 0.998 | 107 min (T4, 26 epochs) |
| MobileNetV2, frozen backbone + new head | 0.916 | 0.934 | 0.988 | 15 min (T4, 10 epochs) |
| MobileNetV2, fine-tuned last 38 layers, Adam 1e-5 | 0.946 | 0.958 | 0.997 | 15 min (T4, 10 epochs) |
| MobileNetV2, fine-tuned last 38 layers, Adam 3e-5 | 0.955 | 0.968 | 0.999 | 14 min (T4, 10 epochs) |
| MobileNetV2, fine-tuned last 38 layers, Adam 1e-4 | 0.9772 | **0.9863** | **0.9995** | 24 min (T4, 17 epochs) |
| **MobileNetV2, whole backbone fine-tuned, Adam 3e-5 (final)** | **0.9774** | 0.9858 | 0.9985 | 39 min (T4, 18 epochs) |

The plan's 10-epoch fine-tuning runs peaked at their last epoch: they were under-trained. With up
to 30 epochs and a higher learning rate or the whole backbone trainable, MobileNetV2 overtakes
the CNN (D-017). The last two runs are tied within noise (val macro-F1 0.97738 vs 0.97722); by
the rule fixed before training (highest val macro-F1) the whole-backbone run is the final model.
Tuning budget: one configuration for the CNN, five for MobileNetV2. Per-run details:
`reports/results.csv`, `reports/runs/`; learning curves: `reports/figures/curves_*.png`.

## Why this project is different

- **Honest evaluation.** PlantVillage contains several photos of the same physical leaf, some
  of them near-identical re-shots. A random split puts them in train and test and inflates the
  score. Here every leaf and every group of near-identical images is in exactly one split,
  checked by automated tests. Known limits are stated openly (see [Data](#data)).
- **Reliability.** Grad-CAM shows what the model looked at; low-confidence inputs get
  "not sure" instead of a confident wrong answer.
- **Engineering.** Config-driven experiments, fixed seeds, tests and CI. GPU training is not
  bit-for-bit deterministic, so a re-run can differ slightly (D-018).

## Quick start

### Google Colab (recommended)

Open `notebooks/00_colab_bootstrap.ipynb` in Colab, select a T4 GPU runtime and run all cells.
It mounts Drive, installs the package, and checks the dataset and the committed splits
(every line should say `PASS`).

### Reproducing the data step

The splits in `splits/` are committed, so training does not need these steps. To rebuild or audit
them:

| Step | Where | What it does |
|---|---|---|
| `python -m plantdx.data.prepare` | any machine, ~1 min | Rebuilds `splits/*.csv` from the official index files and the grouping pairs (`splits/duplicate_pairs.csv`, `splits/sameleaf_pairs.csv`); fails if anything leaks. |
| `notebooks/01_data_audit_eda.ipynb` | Colab, ~10 min | Downloads the images, EDA figures, first (withdrawn) hash audit. |
| `notebooks/02_duplicates_and_resplit.ipynb` | Colab, ~15 min | Searches all images for near-identical pairs, measures each cut-off, writes `splits/duplicate_pairs.csv`, rebuilds and verifies the splits. |
| `notebooks/03_pipeline_check.ipynb` | Colab (T4), ~10 min | Checks the `tf.data` pipeline on the real images, saves augmentation examples, measures speed and memory. |
| `notebooks/04_sameleaf_groups.ipynb` | Colab (T4), ~35 min (~10 with cached embeddings) | Recovers the leaf groups of images without a verified leaf id with CNN embeddings calibrated on verified leaves; writes `splits/sameleaf_pairs.csv` and rebuilds the splits (D-010). |
| `python -m plantdx.data.card` | any machine | Regenerates the class table and the images-per-leaf table in `docs/data_card.md` from `splits/`. |

### Local

```bash
git clone https://github.com/AnarHummatli/plant-disease-detection.git
cd plant-disease-detection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[tf,demo,dev]"
make env     # environment report
make test    # unit tests
```

### Training and baselines

Every experiment is a YAML config that extends `configs/base.yaml`. On Colab, use
`notebooks/05_baselines.ipynb` (CPU is enough), `notebooks/06_train.ipynb` (T4 GPU, one or more
configs in a row), once all training is done `notebooks/07_evaluate.ipynb` (final model and the
one-time test evaluation; no GPU needed) and then `notebooks/08_analysis.ipynb` (error analysis,
Grad-CAM, "not sure" threshold; no GPU needed). They write runs to Drive and export the results for
the repository. From a shell, with the images extracted to `DATA` (the folder that contains
`raw/color/`):

```bash
make baselines DATA=/content/data                          # colour/HOG/LBP + LogReg / linear SVM
make train CONFIG=configs/cnn.yaml DATA=/content/data      # CNN from scratch
make train CONFIG=configs/mobilenetv2_head.yaml DATA=...   # MobileNetV2: frozen backbone, new head
make train CONFIG=configs/mobilenetv2_ft.yaml DATA=...     # then fine-tune from the head (LR 1e-5)
make train CONFIG=configs/mobilenetv2_ft_lr3e-5.yaml DATA=...   # fine-tuning variant (LR 3e-5)
make train CONFIG=configs/mobilenetv2_ft_lr1e-4.yaml DATA=...   # D-017 A: LR 1e-4, up to 30 epochs
make train CONFIG=configs/mobilenetv2_ft_all.yaml DATA=...      # D-017 B: whole backbone, 3e-5
make eval DATA=/content/data                               # select on val, then test ONCE
python -m plantdx.analyze --data-root /content/data        # error analysis, Grad-CAM, "not sure"
```

Model selection uses validation macro-F1 only: the best run of every neural family, and the final
model is the best of those. `plantdx.evaluate` then scores every family winner and the classical
baselines on the test split, once (it refuses a second run), on the full split and on its
leaf-verified part (D-015). Every run writes `config.yaml`, `history.csv`, `best.keras`
(training checkpoint), `inference.keras` (float32, for evaluation and the demo) and
`metrics_val.json` to `<artifacts>/runs/<run_id>/` and one row to the shared results table.

## Project layout

```
configs/          experiment configs (YAML, inherit from base.yaml)
src/plantdx/      all logic: data/ (source, splits, duplicates, sameleaf, pipeline, card), baselines/,
                  models/ (cnn, mobilenet), train, export, evaluate, metrics, results,
                  analysis + analyze (error analysis), gradcam, viz
notebooks/        00 bootstrap, 01 EDA + first audit, 02 duplicate search, 03 pipeline check, 04 leaf groups,
                  05 classical baselines, 06 train neural networks, 07 final model + test evaluation,
                  08 error analysis, Grad-CAM and the "not sure" threshold
splits/           train/val/test CSVs with leaf_id and group_id, grouping pairs (committed)
tests/            pytest: splits, pipeline, baselines, models, trainer, evaluation, analysis, Grad-CAM
app/gradio/       demo
docs/             problem statement, data card, decision log, error analysis (later: report)
reports/figures/  every figure used in the report and this README
```

## Data

[PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage) (Hughes & Salathé, 2015;
Mohanty, Hughes & Salathé, 2016), colour variant, 54,305 images, 38 classes, licence CC BY-SA 3.0.
Images are downloaded by code and are never stored in this repository (`docs/decisions.md`, D-001).
Full description, per-class table and limitations: **[`docs/data_card.md`](docs/data_card.md)**.

| Split | Images | Leaves | Leaf id verified |
|---|---:|---:|---:|
| train | 36,779 | 13,406 | 77% |
| val | 6,121 | 2,236 | 77% |
| test (official, unchanged) | 10,709 | 4,118 | 75% |

What the data checks found (details in `docs/decisions.md`, D-004, D-006 and D-010):

- 24.3% of the images have no entry in the dataset's leaf map, so their leaf is identified only
  by file name. The official split shares 227 such file-name leaf ids between train and test.
- A first duplicate check with a perceptual hash reported 0%, but inspecting the closest pairs
  showed near-identical images it had missed, so that result was withdrawn.
- A correlation search over all 54,305 images (with flips and rotations) found that the images
  crossing the splits are **re-shots of the same leaf**, not plain copies, including in classes
  whose leaf ids come from the official leaf map. Such pairs are grouped before splitting: 43
  cross-split pairs before, 0 after.
- Verified leaves have a median of 4 photos, but file-name ids almost always hold one, so the other
  photos of those leaves were not grouped. CNN embeddings calibrated on verified leaves recover
  only the clearest cases: at 96% precision they find 0.4% of same-leaf pairs. The 565 pairs above
  that cut-off are grouped (253 crossed the splits before, 0 after). This is reported as a
  **negative result**: most hidden same-leaf pairs cannot be recovered from the images.
- In total 696 train images (1.6%) were removed; the official test set is unchanged. Because some
  leakage may remain, test results are also reported on the leaf-verified part of the test set
  (8,033 images, 30 classes), where leakage is excluded by construction.

![One example per class](reports/figures/eda_sample_grid.jpg)

## Disclaimer

This is an educational prototype. It gives preliminary information only, does not replace an
agronomist's diagnosis, and is not intended for safety-critical use.

## License

Code: MIT. Data: see the licences of the respective datasets.
