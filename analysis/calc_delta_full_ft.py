import pandas as pd
import numpy as np

LORA_RESULTS_CSV = "./output/data/lora_abl.csv"
FULLFT_RESULTS_CSV = "./output/data/results_seed101.csv"
LORA_TARGET_RANK = "32.0"


def compute_delta_agg(df: pd.DataFrame, has_rank: bool) -> pd.DataFrame:
    """
    Given a long-format results CSV (base + fine-tuned rows), compute
    deltas vs base and aggregate over seeds.

    If has_rank=True, keeps 'rank' as a grouping key in the aggregated output.
    """
    for col in ["fine_tuned_dataset", "fine_tuned_language"]:
        if col in df.columns:
            df[col] = df[col].replace("n/a", pd.NA)

    # Identify base vs fine-tuned rows
    is_base = df["fine_tuned_dataset"].isna() & df["fine_tuned_language"].isna()
    base_df = df[is_base].copy()
    ft_df = df[~is_base].copy()

    if has_rank:
        if "rank" not in df.columns:
            raise ValueError("has_rank=True but no 'rank' column found in dataframe.")
        ft_df["rank"] = ft_df["rank"].astype(str)
        base_df["rank"] = base_df["rank"].astype(str)

    # Rename base score columns so they don't clash on merge
    base_df = base_df.rename(
        columns={
            "score": "score_base",
            "score_stderr": "score_stderr_base",
            "score_norm": "score_norm_base",
            "score_norm_stderr": "score_norm_stderr_base",
        }
    )

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
            "Missing base rows for some (model, size, seed, eval_dataset, eval_language):\n"
            f"{missing}"
        )

    # Compute per-seed deltas
    merged["delta_score"] = (merged["score"] - merged["score_base"]) * 100
    merged["delta_score_norm"] = (
        merged["score_norm"] - merged["score_norm_base"]
    ) * 100

    # Aggregate over seeds
    agg_keys = [
        "model_name",
        "model_size",
        "fine_tuned_dataset",
        "fine_tuned_language",
        "eval_dataset",
        "eval_language",
    ]
    if has_rank:
        agg_keys.append("rank")

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

    return agg


def main():
    df_lora = pd.read_csv(LORA_RESULTS_CSV)
    df_full = pd.read_csv(FULLFT_RESULTS_CSV)

    delta_lora = compute_delta_agg(df_lora, has_rank=True)
    delta_full = compute_delta_agg(df_full, has_rank=False)

    df_lora_r = delta_lora.copy()

    models = [
        "Llama-3.1-8B-Instruct",
        "Qwen2.5-1.5B-Instruct",
        "gemma-3-4b-it",
    ]
    datasets = ["arc_challenge", "global_mmlu"]
    languages = ["bn", "en", "fr"]

    df_lora_r = df_lora_r[
        df_lora_r["model_name"].isin(models)
        & df_lora_r["fine_tuned_dataset"].isin(datasets)
        & df_lora_r["fine_tuned_language"].isin(languages)
    ].copy()

    delta_full = delta_full[
        delta_full["model_name"].isin(models)
        & delta_full["fine_tuned_dataset"].isin(datasets)
        & delta_full["fine_tuned_language"].isin(languages)
    ].copy()

    cell_keys = [
        "model_name",
        "model_size",
        "fine_tuned_dataset",
        "fine_tuned_language",
        "eval_dataset",
        "eval_language",
    ]

    df_lora_idx = df_lora_r.set_index(cell_keys)
    df_full_idx = delta_full.set_index(cell_keys)

    # Rename columns before join to avoid suffix confusion
    lora_cols = df_lora_idx[["delta_score_mean", "delta_score_norm_mean"]].rename(
        columns={
            "delta_score_mean": "delta_score_mean_lora",
            "delta_score_norm_mean": "delta_score_norm_mean_lora",
        }
    )
    full_cols = df_full_idx[["delta_score_mean", "delta_score_norm_mean"]].rename(
        columns={
            "delta_score_mean": "delta_score_mean_full",
            "delta_score_norm_mean": "delta_score_norm_mean_full",
        }
    )

    comp = lora_cols.join(full_cols, how="inner").reset_index()

    if comp.empty:
        raise ValueError(
            "No overlapping cells between LoRA and FullFT after filtering."
        )

    # Compute correlation, sign agreement, differences
    rho = comp["delta_score_mean_lora"].corr(comp["delta_score_mean_full"])
    rho_norm = comp["delta_score_norm_mean_lora"].corr(
        comp["delta_score_norm_mean_full"]
    )

    sign_agreement = np.mean(
        np.sign(comp["delta_score_mean_lora"]) == np.sign(comp["delta_score_mean_full"])
    )

    comp["delta_diff"] = comp["delta_score_mean_lora"] - comp["delta_score_mean_full"]
    mean_diff = comp["delta_diff"].mean()
    median_diff = comp["delta_diff"].median()

    print("=== LoRA (r=32) vs FullFT: per-cell comparison ===")
    print(f"Number of overlapping cells: {len(comp)}")
    print(f"Correlation (delta_score_mean):        {rho:.4f}")
    print(f"Correlation (delta_score_norm_mean):   {rho_norm:.4f}")
    print(f"Sign agreement (gain/harm):            {sign_agreement * 100:.1f}%")
    print(f"Mean(Δ_lora - Δ_full) [pp]:            {mean_diff:.3f}")
    print(f"Median(Δ_lora - Δ_full) [pp]:          {median_diff:.3f}")


if __name__ == "__main__":
    main()
