import pandas as pd
import numpy as np


def main():
    df = pd.read_csv("./output/data/lora_abl.csv")

    # Identify base vs fine-tuned rows
    is_base = (
        df["fine_tuned_dataset"].isna() | (df["fine_tuned_dataset"] == "n/a")
    ) & (df["fine_tuned_language"].isna() | (df["fine_tuned_language"] == "n/a"))
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

    # Keys that must match between base and fine-tuned (per seed)
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
        validate="many_to_one",  # each (model, size, seed, eval_dataset, eval_language) has one base record
    )

    # Sanity check
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
        "rank",
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

    # Ensure rank is numeric
    agg["rank"] = agg["rank"].astype(float)

    # Rank-level aggregate view
    rank_summary = agg.groupby("rank", as_index=False).agg(
        delta_score_mean_overall=("delta_score_mean", "mean"),
        delta_score_norm_mean_overall=("delta_score_norm_mean", "mean"),
        n_rows=("delta_score_mean", "size"),
    )

    print("=== Rank-level aggregate deltas (percentage points) ===")
    print(rank_summary.sort_values("rank"))

    # Distribution per rank (on delta_score_norm_mean)
    print("\n=== Per-rank distribution of Δ_norm (mean over seeds per cell) ===")
    for r in sorted(agg["rank"].unique()):
        sub = agg[agg["rank"] == r]["delta_score_norm_mean"]
        if sub.empty:
            continue
        stats = {
            "mean": sub.mean(),
            "median": sub.median(),
            "p10": sub.quantile(0.10),
            "p25": sub.quantile(0.25),
            "p75": sub.quantile(0.75),
            "p90": sub.quantile(0.90),
        }
        print(
            f"rank={r:>5}: "
            f"mean={stats['mean']:.3f}, "
            f"median={stats['median']:.3f}, "
            f"p10={stats['p10']:.3f}, "
            f"p25={stats['p25']:.3f}, "
            f"p75={stats['p75']:.3f}, "
            f"p90={stats['p90']:.3f}"
        )

    # Harm rate per rank (fraction of cells with Δ_norm <= -1 pp)
    harm_threshold = 1.0
    harm_by_rank = agg.groupby("rank", as_index=False).agg(
        harm_rate=(
            "delta_score_norm_mean",
            lambda s: float((s <= -harm_threshold).mean()),
        )
    )
    print(f"\n=== Harm rate per rank (Δ_norm_mean <= -{harm_threshold:g} pp) ===")
    print(harm_by_rank.sort_values("rank"))

    # Regime-level view (MT–ML, MT–CL, off-task)
    def classify_regime(row: pd.Series) -> str:
        if row["fine_tuned_dataset"] == row["eval_dataset"]:
            if row["fine_tuned_language"] == row["eval_language"]:
                return "mt_ml"
            else:
                return "mt_cl"
        else:
            return "off_task"

    agg["regime"] = agg.apply(classify_regime, axis=1)

    regime_summary = (
        agg.groupby(["rank", "regime"], as_index=False)
        .agg(
            delta_score_norm_mean_regime=("delta_score_norm_mean", "mean"),
            harm_rate=(
                "delta_score_norm_mean",
                lambda s: float((s <= -harm_threshold).mean()),
            ),
            n_rows=("delta_score_norm_mean", "size"),
        )
        .sort_values(["rank", "regime"])
    )

    print("\n=== Regime-level Δ_norm and harm rates per rank ===")
    print(regime_summary)

    cell_keys = [
        "model_name",
        "model_size",
        "fine_tuned_dataset",
        "fine_tuned_language",
        "eval_dataset",
        "eval_language",
    ]

    def slice_rank(r: float) -> pd.DataFrame:
        df_r = agg[agg["rank"] == r].copy()
        if df_r.empty:
            return df_r
        return df_r.set_index(cell_keys)

    agg32 = slice_rank(32.0)
    agg64 = slice_rank(64.0)
    agg128 = slice_rank(128.0)

    if not agg32.empty and not agg64.empty:
        comp_64_32 = (
            agg32[["delta_score_mean", "delta_score_norm_mean"]]
            .join(
                agg64[["delta_score_mean", "delta_score_norm_mean"]],
                lsuffix="_r32",
                rsuffix="_r64",
                how="inner",
            )
            .reset_index()
        )

        comp_64_32["delta_score_mean_diff_64_minus_32"] = (
            comp_64_32["delta_score_mean_r64"] - comp_64_32["delta_score_mean_r32"]
        )
        comp_64_32["delta_score_norm_mean_diff_64_minus_32"] = (
            comp_64_32["delta_score_norm_mean_r64"]
            - comp_64_32["delta_score_norm_mean_r32"]
        )

        rho_32_64 = comp_64_32["delta_score_mean_r32"].corr(
            comp_64_32["delta_score_mean_r64"]
        )
        rho_32_64_norm = comp_64_32["delta_score_norm_mean_r32"].corr(
            comp_64_32["delta_score_norm_mean_r64"]
        )

        print("\n=== Rank 64 vs 32: per-cell differences ===")
        print(
            f"Number of overlapping cells: {len(comp_64_32)}\n"
            f"Correlation (delta_score_mean, 32 vs 64): {rho_32_64:.4f}\n"
            f"Correlation (delta_score_norm_mean, 32 vs 64): {rho_32_64_norm:.4f}"
        )

    else:
        print("\n[WARN] Missing rank 32 or 64 entries; skipping 64–32 comparison.")

    if not agg32.empty and not agg128.empty:
        comp_128_32 = (
            agg32[["delta_score_mean", "delta_score_norm_mean"]]
            .join(
                agg128[["delta_score_mean", "delta_score_norm_mean"]],
                lsuffix="_r32",
                rsuffix="_r128",
                how="inner",
            )
            .reset_index()
        )

        comp_128_32["delta_score_mean_diff_128_minus_32"] = (
            comp_128_32["delta_score_mean_r128"] - comp_128_32["delta_score_mean_r32"]
        )
        comp_128_32["delta_score_norm_mean_diff_128_minus_32"] = (
            comp_128_32["delta_score_norm_mean_r128"]
            - comp_128_32["delta_score_norm_mean_r32"]
        )

        rho_32_128 = comp_128_32["delta_score_mean_r32"].corr(
            comp_128_32["delta_score_mean_r128"]
        )
        rho_32_128_norm = comp_128_32["delta_score_norm_mean_r32"].corr(
            comp_128_32["delta_score_norm_mean_r128"]
        )

        print("\n=== Rank 128 vs 32: per-cell differences ===")
        print(
            f"Number of overlapping cells: {len(comp_128_32)}\n"
            f"Correlation (delta_score_mean, 32 vs 128): {rho_32_128:.4f}\n"
            f"Correlation (delta_score_norm_mean, 32 vs 128): {rho_32_128_norm:.4f}"
        )

    else:
        print("\n[WARN] Missing rank 32 or 128 entries; skipping 128–32 comparison.")


if __name__ == "__main__":
    main()
