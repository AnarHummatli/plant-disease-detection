# One entry point per stage of the plan.
PY ?= python
CONFIG ?= configs/cnn.yaml
# folder that contains raw/<variant>/ (Colab: notebooks extract the images to /content/data)
DATA ?= /content/data
# final-model candidates (all neural families, selected on val, D-015) and the classical
# baselines; every family winner and every RUNS entry is scored on test once
CANDIDATES ?= cnn_scratch mobilenetv2_head mobilenetv2_ft mobilenetv2_ft_lr3e-5 \
              mobilenetv2_ft_lr1e-4 mobilenetv2_ft_all
RUNS ?= baseline_logreg baseline_linsvm

.PHONY: help install install-local env images data card baselines train eval analyze demo web reproduce test lint format

help:
	@echo "install        core + dev deps (Colab/Kaggle: TensorFlow already present)"
	@echo "install-local  + TensorFlow and Gradio (your own machine)"
	@echo "env            print Python / TF / GPU / artifacts report"
	@echo "images         download the PlantVillage images to DATA (~2.2 GB)        (step 2)"
	@echo "data           rebuild the leaf- and duplicate-grouped splits          (step 2)"
	@echo "card           refresh the class table in docs/data_card.md"
	@echo "baselines      colour/HOG/LBP features + LogReg / linear SVM            (step 3.2)"
	@echo "train          train a model: make train CONFIG=configs/<name>.yaml  (steps 3-4)"
	@echo "eval           select on val, score family winners + RUNS on test ONCE (4.3-4.4)"
	@echo "analyze        error analysis, Grad-CAM, \"not sure\" threshold          (step 5)"
	@echo "demo           launch the Gradio demo (downloads the model once)       (step 6)"
	@echo "web            build the browser demo into build/web (static Space)    (step 6)"
	@echo "reproduce      compare a fresh environment with the reported results (step 7)"
	@echo "test / lint    pytest / ruff"

install:
	$(PY) -m pip install -e ".[dev]"

install-local:
	$(PY) -m pip install -e ".[tf,demo,dev]"

env:
	$(PY) -m plantdx.env

images:
	$(PY) -m plantdx.data.images --out $(DATA)

data:
	$(PY) -m plantdx.data.prepare

card:
	$(PY) -m plantdx.data.card

baselines:
	$(PY) -m plantdx.baselines.run --data-root $(DATA)

train:
	$(PY) -m plantdx.train --config $(CONFIG) --data-root $(DATA)

eval:
	$(PY) -m plantdx.evaluate --data-root $(DATA) --select $(CANDIDATES) --also $(RUNS)

analyze:
	$(PY) -m plantdx.analyze --data-root $(DATA)

demo:
	$(PY) app/gradio/app.py

web:
	$(PY) -m plantdx.web_export --out build/web

reproduce:
	$(PY) -m plantdx.reproduce --data-root $(DATA) --train

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests app
	$(PY) -m ruff format --check src tests app

format:
	$(PY) -m ruff check --fix src tests app
	$(PY) -m ruff format src tests app
