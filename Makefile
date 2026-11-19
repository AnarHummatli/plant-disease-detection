# One entry point per stage of the plan. Targets for later stages print which
# plan step implements them until that step is done.
PY ?= python
CONFIG ?= configs/cnn.yaml
# folder that contains raw/<variant>/ (Colab: notebooks extract the images to /content/data)
DATA ?= /content/data
# final-model candidates (selected on val) and the other families' runs, all scored on test once
CANDIDATES ?= mobilenetv2_head mobilenetv2_ft mobilenetv2_ft_lr3e-5
RUNS ?= baseline_logreg baseline_linsvm cnn_scratch

.PHONY: help install install-local env data card baselines train eval demo test lint format

help:
	@echo "install        core + dev deps (Colab/Kaggle: TensorFlow already present)"
	@echo "install-local  + TensorFlow and Gradio (your own machine)"
	@echo "env            print Python / TF / GPU / artifacts report"
	@echo "data           rebuild the leaf- and duplicate-grouped splits          (step 2)"
	@echo "card           refresh the class table in docs/data_card.md"
	@echo "baselines      colour/HOG/LBP features + LogReg / linear SVM            (step 3.2)"
	@echo "train          train a model: make train CONFIG=configs/<name>.yaml  (steps 3-4)"
	@echo "eval           select the final model, score it and RUNS on test ONCE (steps 4.3-4.4)"
	@echo "demo           launch the Gradio demo                                 (step 6)"
	@echo "test / lint    pytest / ruff"

install:
	$(PY) -m pip install -e ".[dev]"

install-local:
	$(PY) -m pip install -e ".[tf,demo,dev]"

env:
	$(PY) -m plantdx.env

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

demo:
	@echo "Not implemented yet: plan step 6.1 (app/gradio/app.py)"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

format:
	$(PY) -m ruff check --fix src tests
	$(PY) -m ruff format src tests
