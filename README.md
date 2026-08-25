# Donors and Recipients

Official code and result artifact for **“Donors and Recipients: On Asymmetric
Transfer Across Tasks and Languages with Parameter-Efficient Fine-Tuning.”**

We fine-tune nine instruction-tuned language models on one source cell at a
time in a grid of four benchmarks and eleven languages, then evaluate every
ordered source–target pair. The release contains the training and evaluation
code and all 17,424 three-seed result
cells used by the paper's main analysis. Model weights, adapters, raw benchmark
examples, and ablation result files are not redistributed.

## Quick start: reproduce the paper analysis

Python 3.10 or newer is required. The analysis environment is separate from
the GPU training stack.

```console
make install-analysis
make validate-release
make reproduce-analysis
```

The last command is equivalent to:

```console
.venv/bin/python analysis/analysis.py \
  --results results/transfer_results.csv \
  --output-dir outputs/analysis
```

It writes the main figures and machine-readable tables beneath
`outputs/analysis/`. To inspect the available paths and options:

```console
.venv/bin/python analysis/analysis.py --help
```

## Artifact contents

- `results/transfer_results.csv`: the main 9 × 44 × 44 ordered grid,
  aggregated over experiment seeds 17, 42, and 101.
- `analysis/`: the paper-facing statistics, tables, and plotting code,
  including the simplified three-panel Figure 1.
- `dataset_processors/`, `finetune.py`, and `evaluate.py`: training and
  evaluation implementation for the four paper benchmarks.
- `core_grid_sweeps/`: the nine-model, four-benchmark, eleven-language,
  three-experiment-seed sweep definitions.

See [DATASETS.md](DATASETS.md) for dataset provenance, identifiers, sizes,
licences, and access conditions. See [results/README.md](results/README.md) for
the result schema and units.

## Dataset materialisation

Install the full environment and build local copies from the checked-in
manifests:

```console
make install-training
make materialize-datasets
export DONORS_DATA_ROOT="$PWD/data/aligned"
```

`split_benchmarks.py` is local-first: it writes Hugging Face `DatasetDict`
objects under the selected output directory and never publishes by default.
Hub publication requires the explicit `--publish` flag and an account name:

```console
.venv/bin/python split_benchmarks.py \
  --output-dir data/aligned \
  --publish \
  --hf-user YOUR_HF_ACCOUNT
```

## Fine-tuning

Some model repositories are gated; accept their terms and authenticate with
Hugging Face before training. A single run is:

```console
.venv/bin/python finetune.py \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --dataset arc_challenge \
  --language en \
  --experiment_seed 17 \
  --checkpoints_dir outputs/checkpoints \
  --lora_adapters_dir outputs/adapters
```

The `--experiment_seed`
controls model training stochasticity and is one of 17, 42, or 101. The
fine-tuning script performs the paper's fixed 90/10 train/validation partition.

Weights & Biases logging is disabled for direct runs. It is enabled only when
the process is launched by a W&B sweep agent. The sweep commands are:

```console
make finetune-sweep
make eval-base-model-sweep
make eval-ft-model-sweep
```

## Evaluation

Evaluate a base model across the complete 44-task harness grid:

```console
.venv/bin/python evaluate.py \
  --eval_mode base \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --experiment_seed 17 \
  --outputs_dir outputs/evaluation/base
```

Evaluate a saved adapter by temporarily merging it with the base model:

```console
.venv/bin/python evaluate.py \
  --eval_mode adapter \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset arc_challenge \
  --language en \
  --experiment_seed 17 \
  --lora_adapters_dir outputs/adapters \
  --merged_models_dir outputs/merged \
  --outputs_dir outputs/evaluation/fine_tuned
```

Raw harness JSON can be converted and aggregated with
`analysis/parse_lm_eval_output.py` and `analysis/calc_delta.py`. The published
CSV is already aggregated, so those steps are not needed to reproduce the
paper analysis.

## Validation

`make validate-release` checks the exact grid, models, tasks, languages,
regime labels and counts, three-seed coverage, numeric finiteness, duplicate
keys, headline means/positive-transfer rates/harm rates, manifest sizes and
disjointness, and release contents. It exits non-zero on any discrepancy.

With the training dependencies installed and network access available,
`make validate-public-datasets` additionally loads every public benchmark and
language configuration and requires its train/test IDs to match the checked-in
manifests exactly.

## Citation

```bibtex
@inproceedings{dymkiewicz-etal-2026-donors,
  title     = {Donors and Recipients: On Asymmetric Transfer Across Tasks and
               Languages with Parameter-Efficient Fine-Tuning},
  author    = {Dymkiewicz, Kajetan and Vuli\'c, Ivan and Yannakoudakis, Helen and
               Shapira, Eilam and Reichart, Roi and Korhonen, Anna},
  booktitle = {Findings of the Association for Computational Linguistics:
               EMNLP 2026},
  year      = {2026}
}
```

## Licences

Code is released under the [Apache License 2.0](LICENSE). The numerical result
artifact is released separately under [CC BY 4.0](results/LICENSE).
