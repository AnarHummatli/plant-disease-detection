# Plant Disease Detection · Yarpaq Həkimləri

Classifies a leaf photo into one of **38 plant disease / healthy classes** (14 crops) with a
CNN and MobileNetV2 transfer learning, evaluated on a **leaf-grouped, leakage-free split**, with
Grad-CAM explanations, a "not sure" answer for low-confidence inputs, and a Gradio demo.

Neural Networks group project, Azerbaijan State Oil and Industry University, Fall 2026 (topic 06).

> **Status:** stages 1–2 of 8 in progress: project foundation and leakage-checked data splits are
> done; models, results and the demo link will appear here as the next stages are completed.

## Why this project is different

- **Honest evaluation.** PlantVillage often contains several photos of the same physical leaf.
  A random split puts the same leaf in train and test and inflates the score. Here every image
  with a known leaf id has its leaf in exactly one split (checked by automated tests), and a
  perceptual-hash audit confirms that no duplicate image crosses the splits. Known limits are
  stated openly (see [Data](#data)).
- **Reliability.** Grad-CAM shows what the model looked at; low-confidence inputs get
  "not sure" instead of a confident wrong answer.
- **Engineering.** Config-driven experiments, fixed seeds, tests and CI.

## Quick start

### Google Colab (recommended)

Open `notebooks/00_colab_bootstrap.ipynb` in Colab, select a T4 GPU runtime and run all cells.
It mounts Drive, installs the package and checks access to the dataset.

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
src/plantdx/      all logic: data/, models/, train, evaluate, explain
notebooks/        Colab bootstrap, EDA, result plots only
splits/           train/val/test CSVs with leaf_id (committed)
tests/            pytest: splits, pipeline, models
app/gradio/       demo
docs/             problem statement, data card, decision log, report
reports/figures/  every figure used in the report and this README
```

## Data

[PlantVillage](https://huggingface.co/datasets/mohanty/PlantVillage) (Mohanty, Hughes & Salathé,
2016), colour variant, 54,305 images, 38 classes, licence CC BY-SA 3.0. Images are downloaded by
code and are never stored in this repository (`docs/decisions.md`, D-001).

| Split | Images | Leaves | Leaf id verified |
|---|---:|---:|---:|
| train | 37,122 | 13,629 | 76% |
| val | 6,184 | 2,268 | 76% |
| test (official, unchanged) | 10,709 | 4,118 | 75% |

What the data checks found (details in `docs/decisions.md`, D-004):

- 24.3% of the images have no entry in the dataset's leaf map, so their leaf is identified only
  by file name. The official split shares 227 such file-name leaf ids between train and test;
  the 290 train images concerned were removed.
- Perceptual-hash audit: **0% of val/test images have a near-duplicate in an earlier split**. A
  positive control on the removed images finds 5 exact copies, so the check works.
- A hash cannot recognise a *different photo of the same leaf*, so results are also reported on
  the leaf-verified part of the test set (8,033 images).

![One example per class](reports/figures/eda_sample_grid.jpg)

## Disclaimer

This is an educational prototype. It gives preliminary information only, does not replace an
agronomist's diagnosis, and is not intended for safety-critical use.

## License

Code: MIT. Data: see the licences of the respective datasets.
