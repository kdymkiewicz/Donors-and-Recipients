PYTHON ?= python3
VENV ?= .venv
RUN_PYTHON ?= $(VENV)/bin/python
RESULTS ?= results/transfer_results.csv
ANALYSIS_OUTPUT ?= outputs/analysis

.PHONY: install-analysis install-training validate-release \
	validate-public-datasets reproduce-analysis materialize-datasets \
	finetune-sweep eval-base-model-sweep eval-ft-model-sweep

install-analysis:
	$(PYTHON) -m venv $(VENV)
	$(RUN_PYTHON) -m pip install -r requirements-analysis.txt

install-training:
	$(PYTHON) -m venv $(VENV)
	$(RUN_PYTHON) -m pip install -r requirements-training.txt

validate-release:
	$(RUN_PYTHON) scripts/validate_release.py --results $(RESULTS) --manifest-dir split_manifests/canonical

validate-public-datasets:
	$(RUN_PYTHON) scripts/validate_public_datasets.py --manifest-dir split_manifests/canonical

reproduce-analysis:
	$(RUN_PYTHON) analysis/analysis.py --results $(RESULTS) --output-dir $(ANALYSIS_OUTPUT)

materialize-datasets:
	$(RUN_PYTHON) split_benchmarks.py --output-dir data/aligned

finetune-sweep:
	wandb sweep core_grid_sweeps/ft_sweep.yaml

eval-base-model-sweep:
	wandb sweep core_grid_sweeps/eval_base_model_sweep.yaml

eval-ft-model-sweep:
	wandb sweep core_grid_sweeps/eval_ft_model_sweep.yaml
