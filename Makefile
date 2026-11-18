# One entry point per stage of the plan. Targets for later stages print which
# plan step implements them until that step is done.
PY ?= python
CONFIG ?= configs/mobilenetv2_ft.yaml

.PHONY: help install install-local env data train eval demo test lint format

help:
	@echo "install        core + dev deps (Colab/Kaggle: TensorFlow already present)"
	@echo "install-local  + TensorFlow and Gradio (your own machine)"
	@echo "env            print Python / TF / GPU / artifacts report"
	@echo "data           download PlantVillage and build leaf-grouped splits  (step 2)"
	@echo "train          train a model: make train CONFIG=configs/<name>.yaml  (steps 3-4)"
	@echo "eval           evaluate a trained run                                 (step 5)"
	@echo "demo           launch the Gradio demo                                 (step 6)"
	@echo "test / lint    pytest / ruff"

install:
	$(PY) -m pip install -e ".[dev]"

install-local:
	$(PY) -m pip install -e ".[tf,demo,dev]"

env:
	$(PY) -m plantdx.env

data:
	@echo "Not implemented yet: plan step 2.1-2.3 (plantdx.data.prepare)"

train:
	@echo "Not implemented yet: plan steps 3.3 and 4.1 (plantdx.train --config $(CONFIG))"

eval:
	@echo "Not implemented yet: plan step 5 (plantdx.evaluate)"

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
