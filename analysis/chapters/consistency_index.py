# analysis/consistency_index.py
from __future__ import annotations

from itertools import combinations
from typing import List, Dict
import re

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

REGIME_ORDER = ["MT-CL", "CT-ML", "CT-CL"]


def _pick_model_column(df: pd.DataFrame) -> str:
    """Use 'model_name' if present, otherwise fall back to 'model'."""
    if "model_name" in df.columns:
        return "model_name"
    if "model" in df.columns:
        return "model"
    raise KeyError("Expected a 'model_name' or 'model' column in df_agg.")


def _add_regime_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'regime' column:

      MT-CL : same dataset, different language
      CT-ML : different dataset, same language
      CT-CL : different dataset AND different language

    Rows with same dataset & same language (MT-ML) get regime = <NA>.
    """
    df = df.copy()

    same_task = df["fine_tuned_dataset"].astype(str) == df["eval_dataset"].astype(str)
    same_lang = df["fine_tuned_language"].astype(str) == df["eval_language"].astype(str)

    regime = np.select(
        [
            same_task & ~same_lang,
            ~same_task & same_lang,
            ~same_task & ~same_lang,
        ],
        ["MT-CL", "CT-ML", "CT-CL"],
        default="",  # must be str for NumPy 2.x
    )

    df["regime"] = pd.Series(regime, index=df.index, dtype="string").replace("", pd.NA)
    return df


def _prepare_base_df(
    df_agg: pd.DataFrame,
    *,
    delta_col: str,
    drop_same_cell: bool,
) -> pd.DataFrame:
    """Common preprocessing for all CI variants."""
    if delta_col not in df_agg.columns:
        raise KeyError(f"delta_col '{delta_col}' not found in df_agg.")

    df = df_agg.copy()
    df[delta_col] = pd.to_numeric(df[delta_col], errors="coerce")
    df = df.dropna(subset=[delta_col])

    if df.empty:
        raise ValueError("No valid rows after coercing delta_col to numeric.")

    if drop_same_cell:
        # Drop trivial same-task same-language rows (no transfer)
        df = df[
            (df["eval_dataset"] != df["fine_tuned_dataset"])
            | (df["eval_language"] != df["fine_tuned_language"])
        ]

    if df.empty:
        raise ValueError("No non-trivial transfer rows after filtering df_agg.")

    df = _add_regime_column(df)

    # recipient keys
    df["recipient_lang"] = df["eval_language"].astype(str)
    df["recipient_task"] = df["eval_dataset"].astype(str)
    df["recipient_pair"] = (
        df["eval_dataset"].astype(str) + "::" + df["eval_language"].astype(str)
    )

    return df


def _ensure_family_column(
    df: pd.DataFrame, family_col: str = "model_family"
) -> pd.DataFrame:
    """
    Ensure there is a model_family column.

    If missing, infer it from model_name:

      - Special case: all Llama 3.x variants (e.g. 'Llama-3.1-8B-Instruct',
        'Meta-Llama-3.2-1B-Instruct') are mapped to a single family 'Llama-3'.

      - Otherwise, strip the size suffix (e.g. 'Qwen2.5-7B-Instruct' -> 'Qwen2.5').
    """
    if family_col in df.columns:
        return df

    model_col = _pick_model_column(df)

    # Generic pattern: everything up to "-<num>B" (e.g. "Llama-3.1-8B")
    size_pattern = re.compile(r"^(.*?)-\d+(?:\.\d+)?[bB]\b")

    def infer_family(name: str) -> str:
        name_str = str(name)
        lower = name_str.lower()

        # Collapse all Llama 3.x variants into one family
        # Matches 'llama-3', 'llama-3.1', 'meta-llama-3.2', etc.
        if re.search(r"llama-3(?:\.\d+)?", lower):
            return "Llama-3"

        # Generic fallback: strip size suffix if present
        m = size_pattern.search(name_str)
        if m:
            return m.group(1)

        # If nothing matches, just return the full model_name
        return name_str

    df = df.copy()
    df[family_col] = df[model_col].astype(str).map(infer_family)
    return df


def _compute_ci_for_bucket(
    frame: pd.DataFrame,
    *,
    recipient_col: str,
    group_cols: List[str],
    value_col: str,
    model_col: str,
    min_recipients: int = 3,
    min_models: int = 2,
) -> pd.DataFrame:
    """
    Core CI computation.

    For each group defined by `group_cols` (a "source"), we:
      - pivot to (recipient × model) with values from `value_col`
      - require at least `min_recipients` recipients and `min_models` models
      - compute Kendall's τ-b between all model pairs over shared recipients
      - average τ's -> ci_mean_tau
    """
    rows: List[Dict] = []

    for keys, g in frame.groupby(group_cols, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        pivot = g.pivot_table(
            index=recipient_col,
            columns=model_col,
            values=value_col,
            aggfunc="mean",
        )

        # Guards
        if pivot.shape[0] < min_recipients or pivot.shape[1] < min_models:
            continue

        taus: List[float] = []
        pairs_used = 0

        for a, b in combinations(pivot.columns, 2):
            sub = pivot[[a, b]].dropna()
            if sub.shape[0] < min_recipients:
                continue

            tau = kendalltau(sub.iloc[:, 0], sub.iloc[:, 1], variant="b").correlation
            if np.isfinite(tau):
                taus.append(float(tau))
                pairs_used += 1

        n_recipients = int(pivot.shape[0])
        n_models = int(pivot.shape[1])
        n_model_pairs = n_models * (n_models - 1) // 2

        ci = float(np.mean(taus)) if pairs_used else np.nan

        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "n_recipients": n_recipients,
                "n_models": n_models,
                "n_model_pairs": int(n_model_pairs),
                "n_pairs_used": int(pairs_used),
                "ci_mean_tau": ci,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _summary_by_regime(results: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise CI per regime:
      median, 25th/75th percentiles, count of sources.
    """
    if results.empty:
        return pd.DataFrame(columns=["regime", "median", "q1", "q3", "n_sources"])

    summary = (
        results.groupby("regime", dropna=False)["ci_mean_tau"]
        .agg(
            median="median",
            q1=lambda s: s.quantile(0.25),
            q3=lambda s: s.quantile(0.75),
            n_sources="count",
        )
        .reset_index()
    )

    summary["regime"] = pd.Categorical(
        summary["regime"], categories=REGIME_ORDER, ordered=True
    )
    summary = summary.sort_values("regime").reset_index(drop=True)
    return summary


def _run_ci_pipeline(
    df: pd.DataFrame,
    *,
    delta_col: str,
    model_col: str,
    min_recipients: int,
    min_models: int,
) -> pd.DataFrame:
    """
    Common CI pipeline once df is prepared and the model_col is chosen.
    """
    bucket_specs = {
        "MT-CL": ("recipient_lang", ["fine_tuned_dataset", "fine_tuned_language"]),
        "CT-ML": ("recipient_task", ["fine_tuned_dataset", "fine_tuned_language"]),
        "CT-CL": ("recipient_pair", ["fine_tuned_dataset", "fine_tuned_language"]),
    }

    results_list: List[pd.DataFrame] = []
    for regime, (rcol, gcols) in bucket_specs.items():
        sub = df[df["regime"] == regime]
        if sub.empty:
            continue

        res = _compute_ci_for_bucket(
            sub,
            recipient_col=rcol,
            group_cols=gcols,
            value_col=delta_col,
            model_col=model_col,
            min_recipients=min_recipients,
            min_models=min_models,
        )
        if res.empty:
            continue

        res["regime"] = regime
        results_list.append(res)

    results = (
        pd.concat(results_list, ignore_index=True)
        if results_list
        else pd.DataFrame(columns=["regime", "ci_mean_tau"])
    )

    summary = _summary_by_regime(results)
    return summary


def consistency_index(
    df_agg: pd.DataFrame,
    *,
    delta_col: str = "delta_score_norm_mean",
    min_recipients: int = 3,
    min_models: int = 2,
    drop_same_cell: bool = True,
) -> pd.DataFrame:
    """
    Compute Consistency Index (CI) summaries at the dataset level,
    treating each concrete base model (family × size) as a separate model.
    """
    df = _prepare_base_df(df_agg, delta_col=delta_col, drop_same_cell=drop_same_cell)
    model_col = _pick_model_column(df)

    return _run_ci_pipeline(
        df,
        delta_col=delta_col,
        model_col=model_col,
        min_recipients=min_recipients,
        min_models=min_models,
    )


def consistency_index_by_family(
    df_agg: pd.DataFrame,
    *,
    delta_col: str = "delta_score_norm_mean",
    family_col: str = "model_family",
    min_recipients: int = 3,
    min_families: int = 2,
    drop_same_cell: bool = True,
) -> pd.DataFrame:
    """
    Compute CI where "models" are model families.

    Implementation:
      - Aggregate Δ over all sizes within each family for every (source, recipient).
      - Then run the CI pipeline treating `family_col` as the model axis.

    If `family_col` is not present, it is inferred from model_name
    (e.g. 'Llama-3.1-8B-Instruct' -> 'Llama-3').
    """
    df = _prepare_base_df(df_agg, delta_col=delta_col, drop_same_cell=drop_same_cell)
    df = _ensure_family_column(df, family_col=family_col)

    group_cols = [
        family_col,
        "fine_tuned_dataset",
        "fine_tuned_language",
        "eval_dataset",
        "eval_language",
        "regime",
    ]
    df_family = df.groupby(group_cols, as_index=False)[delta_col].mean()

    # Recreate recipient keys (groupby preserved eval_* and regime)
    df_family["recipient_lang"] = df_family["eval_language"].astype(str)
    df_family["recipient_task"] = df_family["eval_dataset"].astype(str)
    df_family["recipient_pair"] = (
        df_family["eval_dataset"].astype(str)
        + "::"
        + df_family["eval_language"].astype(str)
    )

    return _run_ci_pipeline(
        df_family,
        delta_col=delta_col,
        model_col=family_col,
        min_recipients=min_recipients,
        min_models=min_families,
    )


def consistency_index_by_size(
    df_agg: pd.DataFrame,
    *,
    delta_col: str = "delta_score_norm_mean",
    size_col: str = "model_size",
    min_recipients: int = 3,
    min_sizes: int = 2,
    drop_same_cell: bool = True,
) -> pd.DataFrame:
    """
    Compute CI where "models" are size buckets (or exact sizes).

    Implementation:
      - Aggregate Δ over all families within each size for every (source, recipient).
      - Then run the CI pipeline treating `size_col` as the model axis.

    Expects a column `size_col` (default: 'model_size') in df_agg.
    """
    df = _prepare_base_df(df_agg, delta_col=delta_col, drop_same_cell=drop_same_cell)

    if size_col not in df.columns:
        raise KeyError(f"Expected size column '{size_col}' in df_agg.")

    group_cols = [
        size_col,
        "fine_tuned_dataset",
        "fine_tuned_language",
        "eval_dataset",
        "eval_language",
        "regime",
    ]
    df_size = df.groupby(group_cols, as_index=False)[delta_col].mean()

    df_size["recipient_lang"] = df_size["eval_language"].astype(str)
    df_size["recipient_task"] = df_size["eval_dataset"].astype(str)
    df_size["recipient_pair"] = (
        df_size["eval_dataset"].astype(str)
        + "::"
        + df_size["eval_language"].astype(str)
    )

    return _run_ci_pipeline(
        df_size,
        delta_col=delta_col,
        model_col=size_col,
        min_recipients=min_recipients,
        min_models=min_sizes,
    )
