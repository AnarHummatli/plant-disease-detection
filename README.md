# Plant Disease Detection · Yarpaq Həkimləri

Classifies a leaf photo into one of **38 plant disease / healthy classes** (14 crops) with a
CNN and MobileNetV2 transfer learning, evaluated on a **leaf-grouped, leakage-free split**, with
Grad-CAM explanations, a "not sure" answer for low-confidence inputs, and a Gradio demo.

Neural Networks group project, Azerbaijan State Oil and Industry University, Fall 2026 (topic 06).

> **Status:** stage 1 of 8 (project foundation). Results, demo link and figures will appear here
> as the stages are completed.

## Why this project is different

- **Honest evaluation.** PlantVillage often contains several photos of the same physical leaf.
  A random split puts the same leaf in train and test and inflates the score. This project keeps
  every leaf in exactly one split and checks it with automated tests.
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
2016), colour variant, 54,305 images, licence CC BY-SA 3.0. Images are downloaded by code and are
never stored in this repository. See `docs/decisions.md` (D-001) for how the data is loaded.

## Disclaimer

This is an educational prototype. It gives preliminary information only, does not replace an
agronomist's diagnosis, and is not intended for safety-critical use.

## License

Code: MIT. Data: see the licences of the respective datasets.
