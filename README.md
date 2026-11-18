# Plant Disease Detection · Yarpaq Həkimləri

Classifies a leaf photo into one of **38 plant disease / healthy classes** (14 crops) with a
CNN and MobileNetV2 transfer learning, evaluated on a **leaf-grouped, leakage-free split**, with
Grad-CAM explanations, a "not sure" answer for low-confidence inputs, and a Gradio demo.

Neural Networks group project, Azerbaijan State Oil and Industry University, Fall 2026 (topic 06).

> **Status:** stages 1–2 of 8 done (foundation; leakage-checked data, input pipeline, data card).
> Models, results and the demo link will appear here in the next stages.

## Why this project is different

- **Honest evaluation.** PlantVillage contains several photos of the same physical leaf, some
  of them near-identical re-shots. A random split puts them in train and test and inflates the
  score. Here every leaf and every group of near-identical images is in exactly one split,
  checked by automated tests. Known limits are stated openly (see [Data](#data)).
- **Reliability.** Grad-CAM shows what the model looked at; low-confidence inputs get
  "not sure" instead of a confident wrong answer.
- **Engineering.** Config-driven experiments, fixed seeds, tests and CI.

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
| `python -m plantdx.data.prepare` | any machine, ~1 min | Rebuilds `splits/*.csv` from the official index files and `splits/duplicate_pairs.csv`; fails if anything leaks. |
| `notebooks/01_data_audit_eda.ipynb` | Colab, ~10 min | Downloads the images, EDA figures, first (withdrawn) hash audit. |
| `notebooks/02_duplicates_and_resplit.ipynb` | Colab, ~15 min | Searches all images for near-identical pairs, measures each cut-off, writes `splits/duplicate_pairs.csv`, rebuilds and verifies the splits. |
| `notebooks/03_pipeline_check.ipynb` | Colab (T4), ~10 min | Checks the `tf.data` pipeline on the real images, saves augmentation examples, measures speed and memory. |
| `python -m plantdx.data.card` | any machine | Regenerates the class table in `docs/data_card.md` from `splits/`. |

### Local

```bash
git clone https://github.com/AnarHummatli/plant-disease-detection.git
cd plant-disease-detection
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[tf,demo,dev]"
make env     # environment report
make test    # unit tests
```

## Project layout

```
configs/          experiment configs (YAML, inherit from base.yaml)
src/plantdx/      all logic: data/ (source, splits, duplicates, pipeline, card), models/, viz
notebooks/        00 bootstrap, 01 EDA + first audit, 02 duplicate search, 03 pipeline check
splits/           train/val/test CSVs with leaf_id and group_id, duplicate pairs (committed)
tests/            pytest: splits, pipeline, models
app/gradio/       demo
docs/             problem statement, data card, decision log (later: report)
reports/figures/  every figure used in the report and this README
```

## Data

[PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage) (Hughes & Salathé, 2015;
Mohanty, Hughes & Salathé, 2016), colour variant, 54,305 images, 38 classes, licence CC BY-SA 3.0.
Images are downloaded by code and are never stored in this repository (`docs/decisions.md`, D-001).
Full description, per-class table and limitations: **[`docs/data_card.md`](docs/data_card.md)**.

| Split | Images | Leaves | Leaf id verified |
|---|---:|---:|---:|
| train | 36,963 | 13,586 | 76% |
| val | 6,147 | 2,265 | 76% |
| test (official, unchanged) | 10,709 | 4,118 | 75% |

What the data checks found (details in `docs/decisions.md`, D-004 and D-006):

- 24.3% of the images have no entry in the dataset's leaf map, so their leaf is identified only
  by file name. The official split shares 227 such file-name leaf ids between train and test.
- A first duplicate check with a perceptual hash reported 0%, but inspecting the closest pairs
  showed near-identical images it had missed, so that result was withdrawn.
- A correlation search over all 54,305 images (with flips and rotations) found that the images
  crossing the splits are **re-shots of the same leaf**, not plain copies, including in classes
  whose leaf ids come from the official leaf map. Such pairs are grouped before splitting: 43
  cross-split pairs before, 0 after. In total 486 train images (1.1%) were removed; the official
  test set is unchanged.
- No method can recognise every different photo of the same leaf, so results are also reported
  on the leaf-verified part of the test set (8,033 images).

![One example per class](reports/figures/eda_sample_grid.jpg)

## Disclaimer

This is an educational prototype. It gives preliminary information only, does not replace an
agronomist's diagnosis, and is not intended for safety-critical use.

## License

Code: MIT. Data: see the licences of the respective datasets.
