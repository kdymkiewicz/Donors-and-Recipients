# Main-grid transfer results

`transfer_results.csv` contains the complete main experimental grid used by
the paper: 9 models × 44 fine-tuning sources × 44 evaluation targets = 17,424
unique ordered cells. Each row aggregates experiment seeds 17, 42, and 101;
Ablation results are not included.

## Keys and labels

The unique key is:

```text
model_name, fine_tuned_dataset, fine_tuned_language,
eval_dataset, eval_language
```

Datasets are `arc_challenge`, `global_mmlu`, `hellaswag`, and `truthfulqa`.
Languages are `ar`, `bn`, `de`, `en`, `es`, `fr`, `hi`, `id`, `it`, `pt`, and
`zh`. `regime` is derived from the ordered source and target:

| Regime | Source task vs target task | Source language vs target language | Rows |
|---|---|---|---:|
| `MT-ML` | matched | matched | 396 |
| `MT-CL` | matched | cross-language | 3,960 |
| `CT-ML` | cross-task | matched | 1,188 |
| `CT-CL` | cross-task | cross-language | 11,880 |

The task-first labels are canonical throughout this release.

## Schema and units

- Identity: `model_name`, `model_size` (billions of parameters),
  `fine_tuned_dataset`, `fine_tuned_language`, `eval_dataset`,
  `eval_language`, and `regime`.
- Scores: `score_mean`, `score_std`, `score_base_mean`, `score_base_std`,
  `score_norm_mean`, `score_norm_std`, `score_norm_base_mean`, and
  `score_norm_base_std`. Scores are proportions on the unit interval when the
  underlying metric is bounded; `*_norm_*` denotes the harness-normalised
  metric selected for cross-benchmark comparison.
- Deltas: `delta_score_mean`, `delta_score_std`, `delta_score_norm_mean`, and
  `delta_score_norm_std`. Deltas are absolute **percentage points (pp)** and
  equal 100 × (fine-tuned score − corresponding base score).
- Seed coverage: `n_seeds`.
- Standard errors: `score_se`, `score_base_se`, `score_norm_se`,
  `score_norm_base_se`, `delta_score_se`, and `delta_score_norm_se`, computed as
  the across-seed standard deviation divided by √`n_seeds`.

The paper's headline analysis uses `delta_score_norm_mean`, a positive-transfer
criterion of Δ > 0, and a harm criterion of Δ ≤ −1.0 pp. The direct
`MT-ML` source cell is reported separately and excluded from transfer-only
means.

## Licence

The numerical values in `transfer_results.csv` are released under the
[Creative Commons Attribution 4.0 International licence](LICENSE). Cite the
paper and identify modified versions when redistributing the artifact.
