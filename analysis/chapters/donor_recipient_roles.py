# donor_recipient.py
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from typing import Sequence, Tuple
from scipy.stats import spearmanr


from chapters.plot_style import PLOT_FONTS, FIG_SIZE, XLABEL_PAD, YLABEL_PAD


def _aggregate_role(
    df: pd.DataFrame,
    *,
    key_col: str,
    model_col: str = "model_name",
    value_col: str = "delta",
    min_rows_per_model: int = 1,
) -> pd.DataFrame:
    """
    For each (model, key) compute mean delta, then average these per-model
    means across models. Also returns SE across models.
    """
    per_model = (
        df.groupby([model_col, key_col])[value_col]
        .agg(mean_delta="mean", n_rows="count")
        .reset_index()
    )

    per_model = per_model[per_model["n_rows"] >= min_rows_per_model]

    agg = (
        per_model.groupby(key_col)["mean_delta"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "score", "std": "score_std", "count": "n_models"})
        .reset_index()
    )

    agg["score_se"] = agg["score_std"] / np.sqrt(agg["n_models"]).where(
        agg["n_models"] > 1, np.nan
    )
    return agg[[key_col, "score", "score_se", "n_models"]]


def _compute_language_roles_from_agg(
    df_agg: pd.DataFrame,
    *,
    delta_col: str,
    min_rows_per_model: int = 1,
    cross_language_only: bool = True,
    same_dataset_only: bool = True,
) -> pd.DataFrame:
    """
    Donor/recipient roles per language, using seed-aggregated deltas.

    By default this computes roles from the Matched-Task Cross-Language (MT–CL)
    regime: same dataset, different language.
    """
    required = {
        "model_name",
        "fine_tuned_language",
        "eval_language",
        delta_col,
    }
    if same_dataset_only:
        required |= {"fine_tuned_dataset", "eval_dataset"}

    missing = sorted(c for c in required if c not in df_agg.columns)
    if missing:
        raise ValueError(f"Missing columns in df_agg: {missing}")

    dfx = df_agg.copy()
    dfx["delta"] = pd.to_numeric(dfx[delta_col], errors="coerce")

    if same_dataset_only:
        dfx = dfx[dfx["fine_tuned_dataset"] == dfx["eval_dataset"]]

    if cross_language_only:
        dfx = dfx[dfx["fine_tuned_language"] != dfx["eval_language"]]

    donor_df = _aggregate_role(
        dfx,
        key_col="fine_tuned_language",
        min_rows_per_model=min_rows_per_model,
    ).rename(
        columns={
            "fine_tuned_language": "lang",
            "score": "donor",
            "score_se": "donor_se",
            "n_models": "n_models_donor",
        }
    )

    recip_df = _aggregate_role(
        dfx,
        key_col="eval_language",
        min_rows_per_model=min_rows_per_model,
    ).rename(
        columns={
            "eval_language": "lang",
            "score": "recipient",
            "score_se": "recipient_se",
            "n_models": "n_models_recipient",
        }
    )

    merged = pd.merge(donor_df, recip_df, on="lang", how="outer")
    return merged[
        [
            "lang",
            "donor",
            "recipient",
        ]
    ]


def compute_task_roles_from_agg(
    df_agg: pd.DataFrame,
    *,
    delta_col: str,
    min_rows_per_model: int = 1,
    cross_task_only: bool = True,
    same_language_only: bool = True,
) -> pd.DataFrame:
    """
    Donor/recipient roles per task (dataset), using seed-aggregated deltas.

    By default this computes roles from the Cross-Task Matched-Language (CT–ML)
    regime: same language, different task.

    Notes:
      - To get CT–ML, keep same_language_only=True and cross_task_only=True.
    """
    required = {
        "model_name",
        "fine_tuned_dataset",
        "eval_dataset",
        delta_col,
    }
    if same_language_only:
        required |= {"fine_tuned_language", "eval_language"}

    missing = sorted(c for c in required if c not in df_agg.columns)
    if missing:
        raise ValueError(f"Missing columns in df_agg: {missing}")

    dfx = df_agg.copy()
    dfx["delta"] = pd.to_numeric(dfx[delta_col], errors="coerce")

    if same_language_only:
        dfx = dfx[dfx["fine_tuned_language"] == dfx["eval_language"]]

    if cross_task_only:
        dfx = dfx[dfx["fine_tuned_dataset"] != dfx["eval_dataset"]]

    donor_df = _aggregate_role(
        dfx,
        key_col="fine_tuned_dataset",
        min_rows_per_model=min_rows_per_model,
    ).rename(
        columns={
            "fine_tuned_dataset": "dataset",
            "score": "donor",
            "score_se": "donor_se",
            "n_models": "n_models_donor",
        }
    )

    recip_df = _aggregate_role(
        dfx,
        key_col="eval_dataset",
        min_rows_per_model=min_rows_per_model,
    ).rename(
        columns={
            "eval_dataset": "dataset",
            "score": "recipient",
            "score_se": "recipient_se",
            "n_models": "n_models_recipient",
        }
    )

    merged = pd.merge(donor_df, recip_df, on="dataset", how="outer")
    return merged[["dataset", "donor", "recipient"]]


def _plot_roles_scatter(
    roles_df: pd.DataFrame,
    *,
    label_col: str,
    out_path: Path | str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    # ---- style knobs (tweak these) ----
    DOT_SIZE = 60
    DOT_FACE_COLOR = "#1f77b4"
    DOT_EDGE_COLOR = "white"
    DOT_EDGE_WIDTH = 0.8

    AXIS_LABEL_FONTSIZE = PLOT_FONTS["axis_label"]
    TICK_LABEL_FONTSIZE = PLOT_FONTS["tick"]

    POINT_LABEL_FONTSIZE = PLOT_FONTS["point_label"]
    POINT_LABEL_WEIGHT = "bold"
    POINT_LABEL_STROKE_WIDTH = 2.5

    DIAG_LINEWIDTH = 1.2
    DIAG_ALPHA = 0.9

    GRID_LINEWIDTH = 0.5
    GRID_ALPHA = 0.3

    DX_FRAC = 0.018
    DY_FRAC = 0.012

    x = roles_df["donor"].to_numpy(dtype=float)
    y = roles_df["recipient"].to_numpy(dtype=float)

    # symmetric zoom: both axes start at 0.4, square limits
    all_vals = np.concatenate([x, y])
    vmax = float(np.nanmax(all_vals))

    pad = max(vmax * 0.08, 1e-9)
    lo = 0.4
    hi = vmax + pad
    min_span = 0.4
    if hi - lo < min_span:
        hi = lo + min_span

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    # shading: above diagonal (better recipients), below (better donors)
    good_recipient_color = "#d9ead3"  # light green
    good_donor_color = "#cfe2f3"  # light blue

    ax.fill_between(
        [lo, hi], [lo, hi], [lo, lo], facecolor=good_donor_color, alpha=0.18, zorder=0
    )
    ax.fill_between(
        [lo, hi],
        [hi, hi],
        [lo, hi],
        facecolor=good_recipient_color,
        alpha=0.18,
        zorder=0,
    )

    # axes + diagonal
    ax.axhline(0, color="0.3", linewidth=1.0, alpha=0.8, zorder=1)
    ax.axvline(0, color="0.3", linewidth=1.0, alpha=0.8, zorder=1)
    ax.plot(
        [lo, hi],
        [lo, hi],
        linestyle=":",
        color="0.4",
        linewidth=DIAG_LINEWIDTH,
        alpha=DIAG_ALPHA,
        zorder=1.5,
    )

    ax.grid(True, linestyle="--", linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # scatter: blue dots with white outline
    ax.scatter(
        roles_df["donor"],
        roles_df["recipient"],
        s=DOT_SIZE,
        facecolor=DOT_FACE_COLOR,
        edgecolor=DOT_EDGE_COLOR,
        linewidth=DOT_EDGE_WIDTH,
        alpha=0.95,
        zorder=2,
    )

    # label all points, offset from dots
    dx = DX_FRAC * (hi - lo)
    dy = DY_FRAC * (hi - lo)

    stroke = pe.withStroke(
        linewidth=POINT_LABEL_STROKE_WIDTH,
        foreground="white",
    )
    for idx, row in roles_df.reset_index(drop=True).iterrows():
        x0 = float(row["donor"])
        y0 = float(row["recipient"])
        label = str(row[label_col])

        # default vertical jitter: alternate above/below
        direction = 1 if (idx % 2 == 0) else -1
        y_text = y0 + direction * dy

        # horizontal placement & special cases
        if label_col == "lang" and label in {"zh", "en"}:
            x_text = x0 - dx
            ha = "right"
        elif label_col == "lang" and label == "pt":
            x_text = x0 + dx
            y_text = y0 + 2 * dy
            ha = "left"
        else:
            x_text = x0 + dx
            ha = "left"

        ax.text(
            x_text,
            y_text,
            label,
            fontsize=POINT_LABEL_FONTSIZE,
            ha=ha,
            va="center",
            weight=POINT_LABEL_WEIGHT,
            path_effects=[stroke],
            zorder=3,
        )

    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FONTSIZE, labelpad=XLABEL_PAD)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE, labelpad=YLABEL_PAD)
    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_language_donor_recipient(
    df_agg: pd.DataFrame,
    *,
    delta_col: str = "delta_score_norm_mean",
    cross_language_only: bool = True,
    same_dataset_only: bool = True,
    min_rows_per_model: int = 1,
    out_path: Path | str = "./output/plots/language_donor_recipient.png",
) -> pd.DataFrame:
    """
    Convenience wrapper to compute and plot language donor/recipient roles.

    By default uses MT–CL (same dataset, different language).
    """
    roles = _compute_language_roles_from_agg(
        df_agg,
        delta_col=delta_col,
        min_rows_per_model=min_rows_per_model,
        cross_language_only=cross_language_only,
        same_dataset_only=same_dataset_only,
    )
    _plot_roles_scatter(
        roles,
        label_col="lang",
        out_path=out_path,
        title="Language donor vs recipient",
        xlabel="Language Donor Score (pp)",
        ylabel="Language Recipient Score (pp)",
    )
    return roles


def compute_language_roles_mtcl(
    df_agg: pd.DataFrame,
    *,
    delta_col: str = "delta_score_norm_mean",
    min_rows_per_model: int = 1,
) -> pd.DataFrame:
    """
    Convenience wrapper to compute language donor/recipient roles in the
    MT–CL regime (same dataset, different language) from seed-aggregated deltas.

    Returns a DataFrame with columns:
      - lang
      - donor
      - recipient
    """
    return _compute_language_roles_from_agg(
        df_agg,
        delta_col=delta_col,
        min_rows_per_model=min_rows_per_model,
        cross_language_only=True,
        same_dataset_only=True,
    )


LANG_TO_FAMILY = {
    "fr": "Romance",
    "es": "Romance",
    "pt": "Romance",
    "it": "Romance",
    "de": "Germanic",
    "en": "Germanic",
    "hi": "Indic",
    "bn": "Indic",
    "zh": "Chinese",
    "ar": "Arabic",
    "id": "Indonesian",
}


def compute_language_roles_by_family(
    df_agg: pd.DataFrame,
    *,
    delta_col: str = "delta_score_norm_mean",
    min_rows_per_model: int = 1,
    cross_language_only: bool = True,
    same_dataset_only: bool = True,
) -> pd.DataFrame:
    """
    Compute donor/recipient scores per language, then aggregate by family.

    Returns a DataFrame with columns: family, donor, recipient.
    """
    roles = _compute_language_roles_from_agg(
        df_agg,
        delta_col=delta_col,
        min_rows_per_model=min_rows_per_model,
        cross_language_only=cross_language_only,
        same_dataset_only=same_dataset_only,
    )
    roles["family"] = roles["lang"].map(LANG_TO_FAMILY)
    family_roles = (
        roles.groupby("family")[["donor", "recipient"]]
        .mean()
        .reset_index()
    )
    return family_roles


def language_donor_correlations_mtcl(
    df_agg: pd.DataFrame,
    *,
    curated_dataset: str,
    mt_datasets: Sequence[str],
    delta_col: str = "delta_score_norm_mean",
    min_rows_per_model: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """
    Compute MT–CL language donor/recipient roles for:
      - all benchmarks
      - the curated benchmark (e.g. Global-MMLU-Lite)
      - the machine-translated benchmarks (ARC, HellaSwag, TruthfulQA)

    and return the Spearman correlation between donor scores:

      rho_cur_vs_mt : curated vs MT-translated

    Returns
    -------
    roles_all, roles_curated, roles_mt, rho_cur_vs_mt
    """
    # roles for all benchmarks (this is what the main plot uses)
    roles_all = compute_language_roles_mtcl(
        df_agg,
        delta_col=delta_col,
        min_rows_per_model=min_rows_per_model,
    )

    # roles restricted to curated dataset
    roles_curated = compute_language_roles_mtcl(
        df_agg[df_agg["eval_dataset"] == curated_dataset],
        delta_col=delta_col,
        min_rows_per_model=min_rows_per_model,
    )

    # roles restricted to MT datasets
    roles_mt = compute_language_roles_mtcl(
        df_agg[df_agg["eval_dataset"].isin(mt_datasets)],
        delta_col=delta_col,
        min_rows_per_model=min_rows_per_model,
    )

    # Align languages and compute Spearman donor correlation curated vs MT
    don_all = roles_all.set_index("lang")["donor"]
    don_cur = roles_curated.set_index("lang")["donor"]
    don_mt = roles_mt.set_index("lang")["donor"]

    common_cur_mt = don_cur.index.intersection(don_mt.index)

    if len(common_cur_mt) < 2:
        rho_cur_vs_mt = float("nan")
    else:
        rho_cur_vs_mt = spearmanr(
            don_cur.loc[common_cur_mt],
            don_mt.loc[common_cur_mt],
        ).correlation

    return roles_all, roles_curated, roles_mt, rho_cur_vs_mt
