# Aligned datasets

The paper uses four public Hugging Face datasets with one configuration for
each of the eleven languages `ar`, `bn`, `de`, `en`, `es`, `fr`, `hi`, `id`,
`it`, `pt`, and `zh`. Each configuration has `train` and `test` splits.

| Paper name | Public aligned dataset | Upstream sampling source(s) | ID field | Train | Test | Upstream licence / access |
|---|---|---|---|------:|---:|---|
| ARC-Challenge | [Dr4kl3s/arc_challenge_core_grid_seed42](https://huggingface.co/datasets/Dr4kl3s/arc_challenge_core_grid_seed42) | [alexandrainst/m_arc](https://huggingface.co/datasets/alexandrainst/m_arc), `train` / `test` | `id` |   300 | 400 | CC BY-NC 4.0; public, ungated |
| Global-MMLU | [Dr4kl3s/global_mmlu_lite_core_grid_seed42](https://huggingface.co/datasets/Dr4kl3s/global_mmlu_lite_core_grid_seed42) | [CohereLabs/Global-MMLU-Lite](https://huggingface.co/datasets/CohereLabs/Global-MMLU-Lite), `dev` / `test` | `sample_id` |   215 | 400 | Apache 2.0; public, ungated |
| HellaSwag | [Dr4kl3s/hellaswag_coregrid_seed42](https://huggingface.co/datasets/Dr4kl3s/hellaswag_coregrid_seed42) | [alexandrainst/m_hellaswag](https://huggingface.co/datasets/alexandrainst/m_hellaswag) (`ar`–`pt`) and [richmondsin/m_hellaswag](https://huggingface.co/datasets/richmondsin/m_hellaswag) (`zh`), `val` | `id` |   300 | 400 | CC BY-NC 4.0; public, ungated |
| TruthfulQA | [Dr4kl3s/truthfulqa_coregrid_seed42](https://huggingface.co/datasets/Dr4kl3s/truthfulqa_coregrid_seed42) | [alexandrainst/m_truthfulqa](https://huggingface.co/datasets/alexandrainst/m_truthfulqa) (`ar`–`zh`) and [Dr4kl3s/truthfulqa_en_aligned](https://huggingface.co/datasets/Dr4kl3s/truthfulqa_en_aligned) (`en`), `val` / `validation` | `truth_id` (synthetic row ID) |   300 | 400 | translated source: CC BY-NC 4.0; English source follows its upstream TruthfulQA terms; public, ungated |

## Local materialisation and publication

The checked-in canonical manifests freeze the exact train/test IDs shared by
all eleven languages. Materialisation filters by those IDs and performs no
random sampling.

Run:

```console
make install-training
make materialize-datasets
```

## Terms and model access

The repository does not contain third-party benchmark records or model
weights. Dataset and model hosts may change their terms, and gated model
repositories require the user to accept the provider's licence and authenticate
directly. The Apache 2.0 licence at the repository root applies only to this
project's code. It does not supersede dataset or model terms.
