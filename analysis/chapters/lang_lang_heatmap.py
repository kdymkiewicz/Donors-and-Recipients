# language_heatmap.py
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from chapters.plot_style import PLOT_FONTS, FIG_SIZE, XLABEL_PAD, YLABEL_PAD

# Core-grid language order
LANGUAGE_ORDER = [
    "ar",
    "bn",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "id",
    "it",
    "pt",
    "zh",
]


def _build_language_matrix(
    df_agg: pd.DataFrame,
    *,
    delta_col: str,
    lang_order: Sequence[str],
    agg_fn: str = "mean",
) -> Tuple[pd.DataFrame, Sequence[str]]:
    """
    Build a language→language matrix from seed-aggregated deltas.

    Assumes df_agg has already been filtered to the desired regime
    (here: matched-task rows; off-diagonals are MT–CL, diagonal is MT–ML).
    """
    required = {"fine_tuned_language", "eval_language", delta_col}
    missing = sorted(required - set(df_agg.columns))
    if missing:
        raise ValueError(f"df_agg is missing columns: {missing}")

    dfx = df_agg.copy()
    dfx["delta"] = pd.to_numeric(dfx[delta_col], errors="coerce")

    if agg_fn == "mean":
        g = (
            dfx.groupby(["fine_tuned_language", "eval_language"])["delta"]
            .mean()
            .reset_index()
        )
    elif agg_fn == "median":
        g = (
            dfx.groupby(["fine_tuned_language", "eval_language"])["delta"]
            .median()
            .reset_index()
        )
    else:
        raise ValueError("agg_fn must be 'mean' or 'median'.")

    present = sorted(set(g["fine_tuned_language"]) | set(g["eval_language"]))
    order = [l for l in lang_order if l in present] + [
        l for l in present if l not in lang_order
    ]

    A = g.pivot(
        index="fine_tuned_language",
        columns="eval_language",
        values="delta",
    ).reindex(index=order, columns=order)

    return A, order


def plot_language_transfer_heatmap(
    df_agg: pd.DataFrame,
    *,
    delta_col: str = "delta_score_norm_mean",
    agg_fn: str = "mean",
    out_path: str | Path = "./output/plots/language_transfer_heatmap.png",
) -> pd.DataFrame:
    """
    Plot a language→language transfer heatmap from seed-aggregated deltas.

    Regime: matched-task rows (same dataset in fine-tuning and evaluation).
    Off-diagonal entries correspond to Matched-Task Cross-Language (MT–CL)
    transfer; diagonal entries correspond to on-task (MT–ML) and are masked.

    Rows: fine-tuned language (donor)
    Columns: evaluated language (recipient)

    Returns the pivoted matrix.
    """
    # --- filter to matched-task rows (same dataset) ---
    same_task = df_agg["fine_tuned_dataset"] == df_agg["eval_dataset"]
    df_mt = df_agg[same_task].copy()

    if df_mt.empty:
        raise ValueError(
            "No matched-task rows found in df_agg; cannot build language "
            "transfer heatmap."
        )

    A, order = _build_language_matrix(
        df_mt,
        delta_col=delta_col,
        lang_order=LANGUAGE_ORDER,
        agg_fn=agg_fn,
    )

    # colour scale: sequential, with 0 at the bottom
    vmin = float(np.nanmin(A.values))
    vmax = float(np.nanmax(A.values))

    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin == vmax:
        # fallback if something weird happens
        vmin, vmax = 0.0, 1.0

    # if everything is positive, clamp lower bound to 0 so the colourbar starts at 0
    if vmin >= 0.0:
        vmin = 0.0

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap("plasma")

    # darker grey for masked diagonal (clearly "blank", not zero)
    cmap.set_bad("#606060")

    # gray out diagonal by masking it
    A_masked = A.copy()
    for i, lang in enumerate(order):
        A_masked.iloc[i, i] = np.nan

    data = np.ma.masked_invalid(A_masked.values)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    im = ax.imshow(
        data,
        norm=norm,
        cmap=cmap,
        aspect="equal",
    )

    heat_tick_fs = PLOT_FONTS["heatmap_tick"]
    axis_label_fs = PLOT_FONTS["axis_label"]
    cbar_label_fs = PLOT_FONTS["cbar_label"]
    cbar_tick_fs = PLOT_FONTS["cbar_tick"]

    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, ha="center", fontsize=heat_tick_fs)
    ax.set_yticklabels(order, fontsize=heat_tick_fs)

    ax.set_xlabel("Evaluated language", fontsize=axis_label_fs, labelpad=XLABEL_PAD)
    ax.set_ylabel("Fine-tuned language", fontsize=axis_label_fs, labelpad=YLABEL_PAD)

    # subtle grid lines
    n = len(order)
    for k in range(n + 1):
        ax.axhline(k - 0.5, color="white", lw=0.6)
        ax.axvline(k - 0.5, color="white", lw=0.6)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Delta (pp)", fontsize=cbar_label_fs)
    cbar.ax.tick_params(labelsize=cbar_tick_fs)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.12, top=0.98)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    return A
