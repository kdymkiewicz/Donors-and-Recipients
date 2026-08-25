from __future__ import annotations

import numpy as np
import pandas as pd

CONSTRUCTION_TYPE_MAP = {
    "global_mmlu": "human-translated",
    "arc_challenge": "machine-translated",
    "hellaswag": "machine-translated",
    "truthfulqa": "machine-translated",
}


def mtcl_by_construction_type(
    df_agg: pd.DataFrame,
    *,
    delta_col: str = "delta_score_norm_mean",
    harm_threshold: float = 1.0,
    construction_type_map: Mapping[str, str] = CONSTRUCTION_TYPE_MAP,
) -> pd.DataFrame:
    """
    Summarise MT-CL uplift and harm separately for:
      - all MT-CL cells
      - curated benchmarks (e.g. Global-MMLU-Lite)
      - machine-translated benchmarks (ARC, HellaSwag, TruthfulQA)

    Returns a DataFrame indexed by 'subset' with columns:
      - n
      - mean_delta
      - median_delta
      - positive_transfer_rate_pct
      - harm_rate_pct
    """
    df_with_buckets = _assign_bucket(df_agg)

    mtcl = df_with_buckets.loc[df_with_buckets["bucket"] == "MT-CL"].copy()
    if mtcl.empty:
        return pd.DataFrame(
            columns=[
                "subset",
                "n",
                "mean_delta",
                "median_delta",
                "positive_transfer_rate_pct",
                "harm_rate_pct",
            ]
        ).set_index("subset")

    # attach construction type based on eval_dataset (fine_tuned_dataset matches in MT-CL)
    mtcl["construction_type"] = mtcl["eval_dataset"].map(construction_type_map)

    # keep only rows with known construction type
    mtcl = mtcl[mtcl["construction_type"].notna()].copy()
    if mtcl.empty:
        raise ValueError("No MT-CL rows with recognised construction_type.")

    def _stats(sub: pd.DataFrame) -> dict:
        delta = pd.to_numeric(sub[delta_col], errors="coerce").dropna()
        n = int(len(delta))
        if n == 0:
            return {
                "n": 0,
                "mean_delta": np.nan,
                "median_delta": np.nan,
                "positive_transfer_rate_pct": np.nan,
                "harm_rate_pct": np.nan,
            }

        return {
            "n": n,
            "mean_delta": float(delta.mean()),
            "median_delta": float(delta.median()),
            "positive_transfer_rate_pct": float((delta > 0).mean() * 100),
            "harm_rate_pct": float((delta <= -harm_threshold).mean() * 100),
        }

    rows = []

    # all MT-CL (across all benchmarks)
    overall_stats = _stats(mtcl)
    overall_stats["subset"] = "all"
    rows.append(overall_stats)

    # per construction type: curated vs mt
    for ctype, sub in mtcl.groupby("construction_type"):
        stats = _stats(sub)
        stats["subset"] = ctype
        rows.append(stats)

    out = pd.DataFrame(rows).set_index("subset").round(4)
    return out


def uplift(
    df: pd.DataFrame,
    column: str = "delta_score",
    *,
    title_prefix: str = "",
) -> pd.DataFrame:
    """
    Returns a 1-row DataFrame with global uplift stats for `column`.
    """
    delta = df[column].dropna()

    out = pd.DataFrame(
        {
            "mean_delta": [delta.mean()],
            "median_delta": [delta.median()],
            "positive_transfer_rate_pct": [(delta > 0).mean() * 100],
            "n": [int(delta.notna().sum())],
        }
    ).round(4)

    return out


def _assign_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds df['bucket'] in {MT-ML, MT-CL, CT-ML, CT-CL} (or None).

    Assumes df has eval_dataset/eval_language and fine_tuned_dataset/fine_tuned_language.

    Buckets:
      - MT-ML: Matched-Task & Matched-Language (trained cell)
      - MT-CL: Matched-Task, Cross-Language
      - CT-ML: Cross-Task, Matched-Language
      - CT-CL: Cross-Task, Cross-Language
    """
    df = df.copy()
    same_task = df["eval_dataset"].astype(str) == df["fine_tuned_dataset"].astype(str)
    same_lang = df["eval_language"].astype(str) == df["fine_tuned_language"].astype(str)

    # Guard against non-trained rows with 'n/a' etc. being tagged as MT-ML
    is_valid_ft = (
        df["fine_tuned_dataset"].notna()
        & df["fine_tuned_language"].notna()
        & (df["fine_tuned_dataset"].astype(str).str.lower() != "n/a")
        & (df["fine_tuned_language"].astype(str).str.lower() != "n/a")
    )

    df["bucket"] = np.select(
        [
            same_task & same_lang & is_valid_ft,  # MT-ML: trained (task, language) cell
            same_task & ~same_lang,
            ~same_task & same_lang,
            ~same_task & ~same_lang,
        ],
        ["MT-ML", "MT-CL", "CT-ML", "CT-CL"],
        default=None,
    )
    return df


def crosslingual_vs_crosstask(
    df: pd.DataFrame,
    *,
    delta_col: str = "delta_score",
    harm_threshold: float = 1.0,
    df_on_task: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Summarise transfer effects by regime for `delta_col`.

    - `df` should contain only transfer rows (no same (dataset, language) cell),
      i.e., MT-CL, CT-ML, CT-CL.
    - `df_on_task`, if provided, should contain ONLY MT-ML rows
      (same dataset + same language).

    Returns a table with rows:
      - MT-ML (if df_on_task is not None and non-empty)
      - MT-CL, CT-ML, CT-CL
    """
    # Assign buckets for transfer regimes (MT-CL, CT-ML, CT-CL)
    df_with_buckets = _assign_bucket(df)

    rows: list[dict] = []

    # ----- MT-ML (on-task) from df_on_task -----
    if df_on_task is not None and not df_on_task.empty:
        sub = df_on_task[delta_col].dropna()
        n = int(len(sub))
        mean_delta = float(sub.mean()) if n > 0 else np.nan
        median_delta = float(sub.median()) if n > 0 else np.nan
        positive_transfer_rate_pct = (
            float((sub > 0).mean() * 100) if n > 0 else np.nan
        )
        harm_rate_pct = (
            float((sub <= -harm_threshold).mean() * 100) if n > 0 else np.nan
        )

        rows.append(
            {
                "bucket": "MT-ML",
                "n": n,
                "mean_delta": mean_delta,
                "median_delta": median_delta,
                "positive_transfer_rate_pct": positive_transfer_rate_pct,
                "harm_rate_pct": harm_rate_pct,
            }
        )

    # ----- Transfer regimes from df (MT-CL, CT-ML, CT-CL) -----
    bucket_order = ["MT-CL", "CT-ML", "CT-CL"]

    for bucket in bucket_order:
        sub = df_with_buckets.loc[
            df_with_buckets["bucket"] == bucket, delta_col
        ].dropna()

        n = int(len(sub))
        mean_delta = float(sub.mean()) if n > 0 else np.nan
        median_delta = float(sub.median()) if n > 0 else np.nan
        positive_transfer_rate_pct = (
            float((sub > 0).mean() * 100) if n > 0 else np.nan
        )
        harm_rate_pct = (
            float((sub <= -harm_threshold).mean() * 100) if n > 0 else np.nan
        )

        rows.append(
            {
                "bucket": bucket,
                "n": n,
                "mean_delta": mean_delta,
                "median_delta": median_delta,
                "positive_transfer_rate_pct": positive_transfer_rate_pct,
                "harm_rate_pct": harm_rate_pct,
            }
        )

    table = pd.DataFrame(rows).set_index("bucket")
    return table


def _size_bucket(b: float) -> str:
    if pd.isna(b):
        return "Unknown"
    if b <= 1.5:
        return "S (≤1.5B)"
    if b < 7:
        return "M (2–6.9B)"
    return "L (≥7B)"


def _infer_family(df: pd.DataFrame, model_col: str) -> pd.Series:
    """
    Infer model family from the model name using simple substring rules.
    """
    if model_col not in df.columns:
        # Fallback
        return pd.Series("Unknown", index=df.index)

    name = df[model_col].astype(str).str.lower()
    family = np.select(
        [
            name.str.contains("llama"),
            name.str.contains("qwen"),
            name.str.contains("gemma"),
        ],
        ["Llama", "Qwen", "Gemma"],
        default="Other",
    )
    return pd.Series(family, index=df.index)


def harmful_mtcl_breakdown(
    df: pd.DataFrame,
    *,
    delta_col: str = "delta_score_norm_mean",
    harm_threshold: float = 1.0,
    size_billion_col: str | None = None,
    model_col: str | None = "model_name",
) -> pd.DataFrame:
    """
    Decompose harmful MT-CL cases by target language, task, size bucket, and family.

    Parameters
    ----------
    df : pd.DataFrame
        Should contain only transfer rows (no same (dataset, language) cell),
        i.e., MT-CL, CT-ML, CT-CL. Typically, df_agg[~same_cell].
    delta_col : str
        Column with the aggregated delta in percentage points
        (e.g. 'delta_score_norm_mean').
    harm_threshold : float
        Harm threshold in absolute pp. Cells with delta <= -harm_threshold
        are counted as harm.
    size_billion_col : str | None
        Column with model size in billions (e.g. 1.3, 3.1, 8.0). If provided,
        we bucket it into S/M/L using _size_bucket.
    model_col : str | None
        Column with the model name (e.g. 'model_name'). If provided, we infer
        a 'family' label (Llama/Qwen/Gemma/Other) and report by-family stats.

    Returns
    -------
    out : pd.DataFrame
        Multi-indexed by (group, key), with columns:
          - n
          - mean_delta
          - median_delta
          - positive_transfer_rate_pct
          - harm_rate_pct   (Δ <= -harm_threshold)
    """
    # Reuse existing bucket assignment
    df_with_buckets = _assign_bucket(df)

    # Keep only MT-CL rows
    mtcl = df_with_buckets.loc[df_with_buckets["bucket"] == "MT-CL"].copy()

    # Drop rows without a delta
    mtcl = mtcl[mtcl[delta_col].notna()].copy()
    if mtcl.empty:
        return pd.DataFrame(
            columns=[
                "group",
                "key",
                "n",
                "mean_delta",
                "median_delta",
                "positive_transfer_rate_pct",
                "harm_rate_pct",
            ]
        ).set_index(["group", "key"])

    # Mark harm
    mtcl["is_harm"] = mtcl[delta_col] <= -harm_threshold

    rows: list[dict] = []

    def _aggregate(sub: pd.DataFrame, *, group_label: str, key: str) -> None:
        """Append a summary row for a given subset."""
        if sub.empty:
            return

        delta = sub[delta_col]
        n = int(len(delta))

        rows.append(
            {
                "group": group_label,
                "key": key,
                "n": n,
                "mean_delta": float(delta.mean()) if n > 0 else np.nan,
                "median_delta": float(delta.median()) if n > 0 else np.nan,
                "positive_transfer_rate_pct": (
                    float((delta > 0).mean() * 100) if n > 0 else np.nan
                ),
                "harm_rate_pct": (
                    float(sub["is_harm"].mean() * 100) if n > 0 else np.nan
                ),
            }
        )

    # Global summary (all MT-CL cells)
    _aggregate(mtcl, group_label="overall", key="ALL")

    # By target language
    for lang, sub in mtcl.groupby("eval_language"):
        _aggregate(sub, group_label="by_language", key=str(lang))

    # By target dataset / benchmark
    for ds, sub in mtcl.groupby("eval_dataset"):
        _aggregate(sub, group_label="by_dataset", key=str(ds))

    # By size bucket (if numeric size column provided)
    if size_billion_col is not None and size_billion_col in mtcl.columns:
        mtcl = mtcl.copy()
        mtcl["size_bucket"] = mtcl[size_billion_col].apply(_size_bucket)

        for bucket, sub in mtcl.groupby("size_bucket"):
            _aggregate(sub, group_label="by_size_bucket", key=str(bucket))

    # By model family (if model_col provided)
    if model_col is not None:
        mtcl = mtcl.copy()
        mtcl["family"] = _infer_family(mtcl, model_col=model_col)

        for fam, sub in mtcl.groupby("family"):
            _aggregate(sub, group_label="by_family", key=str(fam))

    out = pd.DataFrame(rows)
    out = out.set_index(["group", "key"]).sort_index()
    out = out.round(4)
    return out
