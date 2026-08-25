# task_heatmap.py
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from chapters.plot_style import PLOT_FONTS, FIG_SIZE


# nice labels for the core-grid tasks
TASK_LABEL = {
    "arc_challenge": "ARC-Challenge",
    "global_mmlu": "Global MMLU",
    "truthfulqa": "TruthfulQA",
    "hellaswag": "HellaSwag",
}

# desired order on both axes
TASK_ORDER = [
    "arc_challenge",
    "truthfulqa",
    "hellaswag",
    "global_mmlu",
]


def _build_task_matrix(
    df_agg: pd.DataFrame,
    *,
    delta_col: str,
    task_order: Sequence[str],
) -> Tuple[pd.DataFrame, Sequence[str]]:
    """
    Build a task→task matrix from seed-aggregated deltas.

    Assumes df_agg has already been filtered to the desired regime
    (e.g., cross-task, matched-language for CT–ML).
    """
    required = {"fine_tuned_dataset", "eval_dataset", delta_col}
    missing = sorted(required - set(df_agg.columns))
    if missing:
        raise ValueError(f"df_agg is missing columns: {missing}")

    dfx = df_agg.copy()
    dfx["delta"] = pd.to_numeric(dfx[delta_col], errors="coerce")

    # mean over models + languages + seeds for each (source_task, target_task)
    g = (
        dfx.groupby(["fine_tuned_dataset", "eval_dataset"])["delta"]
        .mean()
        .reset_index()
    )

    # tasks actually present
    present = sorted(set(g["fine_tuned_dataset"]) | set(g["eval_dataset"]))
    order = [t for t in task_order if t in present] + [
        t for t in present if t not in task_order
    ]

    A = g.pivot(
        index="fine_tuned_dataset",
        columns="eval_dataset",
        values="delta",
    ).reindex(index=order, columns=order)

    # mask the diagonal (on-task), even though CT–ML should already exclude it
    for t in order:
        if t in A.columns:
            A.loc[t, t] = np.nan

    return A, order


def plot_task_transfer_heatmap(
    df_agg: pd.DataFrame,
    *,
    delta_col: str = "delta_score_norm_mean",
    out_path: str | Path = "./output/plots/task_transfer_heatmap.png",
    annotate_threshold: float | None = 1.0,
) -> pd.DataFrame:
    """
    Plot a task→task transfer heatmap from seed-aggregated deltas.

    Regime: cross-task, matched-language (CT–ML) only.

    Rows: fine-tuned task (donor)
    Columns: evaluated task (recipient)

    Returns the pivoted matrix.
    """
    # --- filter to CT–ML: different task, same language ---
    same_language = df_agg["fine_tuned_language"] == df_agg["eval_language"]
    diff_task = df_agg["fine_tuned_dataset"] != df_agg["eval_dataset"]
    df_ctml = df_agg[same_language & diff_task].copy()

    if df_ctml.empty:
        raise ValueError(
            "No cross-task, matched-language (CT–ML) rows found in df_agg; "
            "cannot build task transfer heatmap."
        )

    A, order = _build_task_matrix(
        df_ctml,
        delta_col=delta_col,
        task_order=TASK_ORDER,
    )

    # symmetric colour limits around 0
    vmax = (
        float(np.nanmax(np.abs(A.values)))
        if np.isfinite(np.nanmax(np.abs(A.values)))
        else 1.0
    )

    cmap = LinearSegmentedColormap.from_list(
        "red_white_green",
        ["#c4302b", "#ffffff", "#1b9e77"],
        N=256,
    )
    cmap.set_bad("#606060")  # diagonal / NaNs

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    im = ax.imshow(
        A.values,
        vmin=-vmax,
        vmax=vmax,
        cmap=cmap,
        aspect="equal",
    )

    labels = [TASK_LABEL.get(t, t) for t in order]

    heat_tick_fs = PLOT_FONTS["heatmap_tick"]
    axis_label_fs = PLOT_FONTS["axis_label"]
    annot_fs = PLOT_FONTS["heatmap_annot"]
    cbar_label_fs = PLOT_FONTS["cbar_label"]
    cbar_tick_fs = PLOT_FONTS["cbar_tick"]

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=heat_tick_fs)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=heat_tick_fs)

    n = len(order)
    for k in range(n + 1):
        ax.axhline(k - 0.5, color="white", lw=0.8)
        ax.axvline(k - 0.5, color="white", lw=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xlabel("Evaluated task (recipient)", fontsize=axis_label_fs)
    ax.set_ylabel("Fine-tuned task (donor)", fontsize=axis_label_fs)

    if annotate_threshold is not None and annotate_threshold >= 0:
        for i in range(n):
            for j in range(n):
                val = A.iat[i, j]
                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.1f}",
                        ha="center",
                        va="center",
                        fontsize=annot_fs,
                        color="black",
                    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Delta (pp)", fontsize=cbar_label_fs)
    cbar.ax.tick_params(labelsize=cbar_tick_fs)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout(pad=0.6)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.22, top=0.98)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    return A
