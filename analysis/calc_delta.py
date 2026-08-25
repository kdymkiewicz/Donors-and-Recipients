import pandas as pd
import numpy as np


def main():
    df_seed42 = pd.read_csv("./output/data/results_seed42.csv")
    df_seed17 = pd.read_csv("./output/data/results_seed17.csv")
    df_seed101 = pd.read_csv("./output/data/results_seed101.csv")

    df = pd.concat([df_seed17, df_seed42, df_seed101], ignore_index=True)

    # Identify base vs fine-tuned rows
    is_base = (df["fine_tuned_dataset"].isna()) & (df["fine_tuned_language"].isna())
    base_df = df[is_base].copy()
    ft_df = df[~is_base].copy()

    # Rename base score columns so they don't clash on merge
    base_df = base_df.rename(
        columns={
            "score": "score_base",
            "score_stderr": "score_stderr_base",
            "score_norm": "score_norm_base",
            "score_norm_stderr": "score_norm_stderr_base",
        }
    )

    # Keys that must match between base and finetuned (per seed)
    merge_keys = ["model_name", "model_size", "seed", "eval_dataset", "eval_language"]

    # Merge fine-tuned rows with the corresponding base row
    merged = ft_df.merge(
        base_df[
            merge_keys
            + [
                "score_base",
                "score_stderr_base",
                "score_norm_base",
                "score_norm_stderr_base",
            ]
        ],
        on=merge_keys,
        how="left",
        validate="many_to_one",
    )

    if merged["score_base"].isna().any():
        missing = merged[merged["score_base"].isna()][merge_keys].drop_duplicates()
        raise ValueError(
            f"Missing base rows for some (model, size, seed, eval_dataset, eval_language):\n{missing}"
        )

    # Compute per-seed deltas
    merged["delta_score"] = (merged["score"] - merged["score_base"]) * 100
    merged["delta_score_norm"] = (
        merged["score_norm"] - merged["score_norm_base"]
    ) * 100

    merged.to_csv("./output/data/delta_per_seed.csv", index=False)

    # Aggregate over seeds
    agg_keys = [
        "model_name",
        "model_size",
        "fine_tuned_dataset",
        "fine_tuned_language",
        "eval_dataset",
        "eval_language",
    ]

    agg = merged.groupby(agg_keys, as_index=False).agg(
        score_mean=("score", "mean"),
        score_std=("score", "std"),
        score_base_mean=("score_base", "mean"),
        score_base_std=("score_base", "std"),
        score_norm_mean=("score_norm", "mean"),
        score_norm_std=("score_norm", "std"),
        score_norm_base_mean=("score_norm_base", "mean"),
        score_norm_base_std=("score_norm_base", "std"),
        delta_score_mean=("delta_score", "mean"),
        delta_score_std=("delta_score", "std"),
        delta_score_norm_mean=("delta_score_norm", "mean"),
        delta_score_norm_std=("delta_score_norm", "std"),
        n_seeds=("seed", "nunique"),
    )

    # Standard errors
    for col in [
        "score",
        "score_base",
        "score_norm",
        "score_norm_base",
        "delta_score",
        "delta_score_norm",
    ]:
        std_col = f"{col}_std"
        se_col = f"{col}_se"
        if std_col in agg.columns:
            agg[se_col] = agg[std_col] / np.sqrt(agg["n_seeds"])

    agg.to_csv("./output/data/delta_agg.csv", index=False)


if __name__ == "__main__":
    main()
