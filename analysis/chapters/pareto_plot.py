# pareto_on_off.py
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chapters.plot_style import PLOT_FONTS, FIG_SIZE, XLABEL_PAD, YLABEL_PAD


# ---------- pretty dataset labels (used for plot legend + returned tables) ----------
DATASET_LABEL_MAP: dict[str, str] = {
    "arc_challenge": "ARC-Challenge",
    "hellaswag": "HellaSwag",
    "truthfulqa": "TruthfulQA",
    "global_mmlu": "Global-MMLU",
}

PARETO_X_LABEL = "MT-CL gain (Δ pp)"
PARETO_Y_LABEL = "Mean off-task Δ (pp)"

LEGEND_X = 1.02
LEGEND_Y_DATASETS = 1.00
LEGEND_Y_FAMILY   = 0.63
LEGEND_Y_SIZE     = 0.33

FAMILY_PANEL_TITLES: dict[str, str] = {
    "Llama": "Llama 3",
    "Qwen": "Qwen 2.5",
    "Gemma": "Gemma 3",
}

TASK_STYLES: dict[str, tuple[str, str]] = {
    "arc_challenge": ("#0072B2", "o"),
    "global_mmlu": ("#E69F00", "s"),
    "hellaswag": ("#009E73", "^"),
    "truthfulqa": ("#D55E00", "D"),
}


def _size_bucket(b: float) -> str:
    if pd.isna(b):
        return "Unknown"
    if b <= 1.5:
        return "S (≤1.5B)"
    if b < 7:
        return "M (2–6.9B)"
    return "L (≥7B)"


def _add_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Add model family + size bucket."""
    dfx = df.copy()

    name = dfx["model_name"].astype(str).str.lower()
    dfx["family"] = np.select(
        [
            name.str.contains("llama"),
            name.str.contains("qwen"),
            name.str.contains("gemma"),
        ],
        ["Llama", "Qwen", "Gemma"],
        default="Other",
    )

    dfx["size_bucket"] = pd.to_numeric(dfx["model_size"], errors="coerce").apply(
        _size_bucket
    )
    return dfx


def _check_required(df: pd.DataFrame, required: set[str], df_name: str = "df") -> None:
    missing = sorted(c for c in required if c not in df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {df_name}: {missing}")


def _prepare_delta(df: pd.DataFrame, delta_col: str) -> pd.DataFrame:
    """Return a copy with a numeric 'delta' column."""
    dfx = df.copy()
    dfx["delta"] = pd.to_numeric(dfx[delta_col], errors="coerce")
    return dfx

def _compute_on_off(
    df: pd.DataFrame,
    *,
    delta_col: str,
    strict_on_lang: bool,
    harm_threshold: float,
    min_n_on: int,
    min_n_off: int,
) -> pd.DataFrame:
    """
    For each (model, source dataset/lang), compute:
      x = mean on-target Δ (matched-task, cross-language by default)
      y = mean off-task Δ (all other datasets)
    """
    dfx = _prepare_delta(df, delta_col)

    same_task = dfx["eval_dataset"].astype(str) == dfx["fine_tuned_dataset"].astype(str)
    same_lang = dfx["eval_language"].astype(str) == dfx["fine_tuned_language"].astype(
        str
    )

    if strict_on_lang:
        on = same_task & same_lang
    else:
        # matched-task, cross-language
        on = same_task & ~same_lang

    off = dfx["eval_dataset"].astype(str) != dfx["fine_tuned_dataset"].astype(str)

    group_keys: list[str] = [
        "model_name",
        "model_size",
        "family",
        "size_bucket",
        "fine_tuned_dataset",
        "fine_tuned_language",
    ]

    on_tbl = (
        dfx[on]
        .groupby(group_keys, dropna=False)["delta"]
        .agg(x="mean", n_on="count")
        .reset_index()
    )

    off_tbl = (
        dfx[off]
        .groupby(group_keys, dropna=False)["delta"]
        .agg(
            y="mean",
            n_off="count",
            harm_rate=lambda s: (s <= -float(harm_threshold)).mean() * 100.0,
        )
        .reset_index()
    )

    out = on_tbl.merge(off_tbl, on=group_keys, how="inner")
    out = out[(out["n_on"] >= min_n_on) & (out["n_off"] >= min_n_off)].copy()
    return out


def _plot_pareto(tbl: pd.DataFrame, out_path: Path | str) -> None:
    out_path = Path(out_path)

    label_fontsize = PLOT_FONTS["axis_label"]
    tick_fontsize = PLOT_FONTS["tick"]
    legend_fontsize = PLOT_FONTS["legend"]
    legend_title_fontsize = PLOT_FONTS["legend_title"]

    # colour by fine_tuned_dataset, marker by family, size by size_bucket
    datasets = sorted(tbl["fine_tuned_dataset"].astype(str).unique())
    families = sorted(tbl["family"].astype(str).unique())
    size_buckets = ["S (≤1.5B)", "M (2–6.9B)", "L (≥7B)"]

    color_cycle = (
        plt.rcParams["axes.prop_cycle"]
        .by_key()
        .get("color", ["C0", "C1", "C2", "C3", "C4", "C5"])
    )
    color_map = {d: color_cycle[i % len(color_cycle)] for i, d in enumerate(datasets)}
    marker_map = {f: m for f, m in zip(families, ["D", "s", "^", "v", "P", "X"])}
    size_map = {"S (≤1.5B)": 25, "M (2–6.9B)": 75, "L (≥7B)": 125}

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for _, r in tbl.iterrows():
        c = color_map.get(r["fine_tuned_dataset"], "C0")
        m = marker_map.get(r["family"], "o")
        s = size_map.get(r["size_bucket"], 60)

        ax.scatter(
            r["x"],
            r["y"],
            s=s,
            marker=m,
            color=c,
            edgecolor="k",
            linewidth=0.5,
            alpha=0.9,
            zorder=3,
        )

    ax.axvline(0.0, ls="--", lw=1)
    ax.axhline(0.0, ls="--", lw=1)

    x_min, x_max = ax.get_xlim()
    ax.set_xlim(left=-6, right=x_max)

    ax.set_xlabel(PARETO_X_LABEL, fontsize=label_fontsize,labelpad=XLABEL_PAD)
    ax.set_ylabel(PARETO_Y_LABEL, fontsize=label_fontsize)

    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    dataset_handles = [
        plt.Line2D([0], [0], marker="o", ls="none", color=color_map[d], markersize=8)
        for d in datasets
    ]
    dataset_labels = [DATASET_LABEL_MAP.get(d, d) for d in datasets]

    legend_datasets = ax.legend(
        dataset_handles,
        dataset_labels,
        title="Fine-tuned Dataset",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(LEGEND_X, LEGEND_Y_DATASETS),
        borderaxespad=0.0,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
    )
    ax.add_artist(legend_datasets)

    fam_handles = [
        plt.Line2D(
            [0], [0], marker=marker_map[f], ls="none", color="black", markersize=8
        )
        for f in families
    ]
    legend_fams = ax.legend(
        fam_handles,
        families,
        title="Model Family",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(LEGEND_X, LEGEND_Y_FAMILY),
        borderaxespad=0.0,
        fontsize=legend_fontsize,
        title_fontsize=legend_title_fontsize,
    )
    ax.add_artist(legend_fams)

    size_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            ls="none",
            color="gray",
            markersize=np.sqrt(size_map[b] / np.pi),
        )
        for b in size_buckets
        if b in tbl["size_bucket"].values
    ]
    if size_handles:
        legend_sizes = ax.legend(
            size_handles,
            [b for b in size_buckets if b in tbl["size_bucket"].values],
            title="Model Size",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(LEGEND_X, LEGEND_Y_SIZE),
            borderaxespad=0.0,
            fontsize=legend_fontsize,
            title_fontsize=legend_title_fontsize,
        )
        ax.add_artist(legend_sizes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 0.71, 1.0))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _summary_marker_area(size_billion: float) -> float:
    """Map parameter count to a visually distinct, monotonically increasing area."""
    return 14.0 + 14.0 * size_billion**1.2


def _plot_pareto_family_summary(tbl: pd.DataFrame, out_path: Path | str) -> None:
    """Plot family panels with faint language runs and model--task means."""
    out_path = Path(out_path)
    plot_tbl = tbl.copy()
    plot_tbl["model_size"] = pd.to_numeric(plot_tbl["model_size"], errors="coerce")

    summary = (
        plot_tbl.groupby(
            ["model_name", "model_size", "family", "fine_tuned_dataset"],
            as_index=False,
        )[["x", "y"]]
        .mean()
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.15, 2.95),
        sharex=True,
        sharey=True,
    )

    for ax, family in zip(axes, FAMILY_PANEL_TITLES, strict=True):
        family_raw = plot_tbl[plot_tbl["family"] == family]
        family_summary = summary[summary["family"] == family]

        for task, (colour, marker) in TASK_STYLES.items():
            raw_task = family_raw[family_raw["fine_tuned_dataset"] == task]
            ax.scatter(
                raw_task["x"],
                raw_task["y"],
                s=9,
                marker=marker,
                facecolor=colour,
                edgecolor="none",
                alpha=0.20,
                zorder=2,
            )

        # Draw larger models first so smaller markers remain visible on top.
        sizes_descending = sorted(
            family_summary["model_size"].dropna().unique(), reverse=True
        )
        for layer, size_billion in enumerate(sizes_descending, start=4):
            size_data = family_summary[
                family_summary["model_size"] == size_billion
            ]
            for task, (colour, marker) in TASK_STYLES.items():
                task_data = size_data[size_data["fine_tuned_dataset"] == task]
                ax.scatter(
                    task_data["x"],
                    task_data["y"],
                    s=_summary_marker_area(float(size_billion)),
                    marker=marker,
                    facecolor=colour,
                    edgecolor="#202020",
                    linewidth=0.75,
                    alpha=0.9,
                    zorder=layer,
                )

        ax.axvline(
            0,
            color="#6b7280",
            linestyle=(0, (3, 2)),
            linewidth=0.8,
            zorder=1,
        )
        ax.axhline(
            0,
            color="#6b7280",
            linestyle=(0, (3, 2)),
            linewidth=0.8,
            zorder=1,
        )
        ax.grid(color="#d1d5db", linewidth=0.55, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(
            FAMILY_PANEL_TITLES[family], fontsize=9.2, fontweight="semibold", pad=4
        )
        ax.tick_params(axis="both", labelsize=7.5, length=2.5, width=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    axes[0].set_xlim(-3.25, 8.15)
    axes[0].set_ylim(-1.25, 6.65)
    axes[0].set_ylabel("Mean off-task transfer (pp)", fontsize=9, labelpad=4)
    fig.supxlabel("Mean MT–CL transfer (pp)", fontsize=9, y=0.225)

    task_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="none",
            markerfacecolor=colour,
            markeredgecolor="#202020",
            markeredgewidth=0.6,
            markersize=5.5,
            label=DATASET_LABEL_MAP[task],
        )
        for task, (colour, marker) in TASK_STYLES.items()
    ]
    parameter_sizes = sorted(plot_tbl["model_size"].dropna().unique())
    size_handles = [
        plt.scatter(
            [],
            [],
            s=_summary_marker_area(float(size)),
            facecolor="#9ca3af",
            edgecolor="#202020",
            linewidth=0.6,
            label=f"{size:g}B",
        )
        for size in parameter_sizes
    ]

    task_legend = fig.legend(
        handles=task_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.095),
        ncol=4,
        frameon=False,
        fontsize=7.1,
        handletextpad=0.3,
        columnspacing=0.8,
    )
    size_legend = fig.legend(
        handles=size_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.008),
        ncol=len(size_handles),
        frameon=False,
        fontsize=6.9,
        handletextpad=0.2,
        columnspacing=0.75,
    )
    task_heading = fig.text(
        0.0, 0.132, "Source task:", fontsize=7.2, fontweight="semibold"
    )
    size_heading = fig.text(
        0.0, 0.045, "Parameters:", fontsize=7.2, fontweight="semibold"
    )

    fig.subplots_adjust(left=0.10, right=0.995, top=0.91, bottom=0.38, wspace=0.14)

    # Centre the source-task row, then align the parameters row to its left edge.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap = 0.018
    task_heading_width = task_heading.get_window_extent(renderer).transformed(
        fig.transFigure.inverted()
    ).width
    task_legend_width = task_legend.get_window_extent(renderer).transformed(
        fig.transFigure.inverted()
    ).width
    footer_start_x = 0.5 - (task_heading_width + gap + task_legend_width) / 2
    task_heading.set_position((footer_start_x, 0.132))
    task_legend.set_bbox_to_anchor(
        (footer_start_x + task_heading_width + gap, 0.095),
        transform=fig.transFigure,
    )

    size_heading_width = size_heading.get_window_extent(renderer).transformed(
        fig.transFigure.inverted()
    ).width
    size_heading.set_position((footer_start_x, 0.045))
    size_legend.set_bbox_to_anchor(
        (footer_start_x + size_heading_width + gap, 0.008),
        transform=fig.transFigure,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_pareto_on_off(
    df_agg: pd.DataFrame,
    *,
    delta_col: str = "delta_score_norm_mean",
    strict_on_lang: bool = False,
    harm_threshold: float = 1.0,
    min_n_on: int = 3,
    min_n_off: int = 5,
    out_path: Path | str = "analysis/output/plots/pareto_on_off.png",
    return_full: bool = False,
    format_dataset_labels: bool = True,
) -> pd.DataFrame:
    """
    Build and save a Pareto plot:
        x = mean on-target Δ (matched-task, cross-language by default)
        y = mean off-task Δ (all other datasets)

    Returns a table used for plotting.

    By default this returns only:
      ['model_name','fine_tuned_dataset','fine_tuned_language', PARETO_X_LABEL, PARETO_Y_LABEL]
    and fine_tuned_dataset is formatted via DATASET_LABEL_MAP.

    Set return_full=True to return the full debug table (also with renamed columns).
    """
    required = {
        "model_name",
        "model_size",
        "fine_tuned_dataset",
        "fine_tuned_language",
        "eval_dataset",
        "eval_language",
        delta_col,
    }
    _check_required(df_agg, required, df_name="df_agg")

    df_meta = _add_meta(df_agg)
    tbl = _compute_on_off(
        df_meta,
        delta_col=delta_col,
        strict_on_lang=strict_on_lang,
        harm_threshold=harm_threshold,
        min_n_on=min_n_on,
        min_n_off=min_n_off,
    )

    _plot_pareto(tbl, out_path=out_path)

    out = tbl.copy()

    if format_dataset_labels:
        out["fine_tuned_dataset"] = (
            out["fine_tuned_dataset"]
            .astype(str)
            .map(DATASET_LABEL_MAP)
            .fillna(out["fine_tuned_dataset"].astype(str))
        )

    out = out.rename(columns={"x": PARETO_X_LABEL, "y": PARETO_Y_LABEL})

    if return_full:
        return out

    return out[
        [
            "model_name",
            "fine_tuned_dataset",
            "fine_tuned_language",
            PARETO_X_LABEL,
            PARETO_Y_LABEL,
        ]
    ].copy()


def plot_pareto_family_summary(
    df_agg: pd.DataFrame,
    *,
    delta_col: str = "delta_score_norm_mean",
    strict_on_lang: bool = False,
    harm_threshold: float = 1.0,
    min_n_on: int = 3,
    min_n_off: int = 5,
    out_path: Path | str = "analysis/output/plots/pareto_on_off_summary.png",
) -> pd.DataFrame:
    """Create the summary with one panel per model family."""
    required = {
        "model_name",
        "model_size",
        "fine_tuned_dataset",
        "fine_tuned_language",
        "eval_dataset",
        "eval_language",
        delta_col,
    }
    _check_required(df_agg, required, df_name="df_agg")

    tbl = _compute_on_off(
        _add_meta(df_agg),
        delta_col=delta_col,
        strict_on_lang=strict_on_lang,
        harm_threshold=harm_threshold,
        min_n_on=min_n_on,
        min_n_off=min_n_off,
    )
    _plot_pareto_family_summary(tbl, out_path=out_path)

    return (
        tbl.groupby(
            ["model_name", "model_size", "family", "fine_tuned_dataset"],
            as_index=False,
        )[["x", "y"]]
        .mean()
        .rename(columns={"x": PARETO_X_LABEL, "y": PARETO_Y_LABEL})
    )


def compute_task_on_off_summary(
    df: pd.DataFrame,
    *,
    delta_col: str = "delta_score_norm_mean",
    harm_threshold: float = 1.0,
    task_col: str = "fine_tuned_dataset",
    ft_lang_col: str = "fine_tuned_language",
    eval_task_col: str = "eval_dataset",
    eval_lang_col: str = "eval_language",
) -> pd.DataFrame:
    """
    Per-task on-task vs off-task statistics, consistent with plot_pareto_on_off
    (strict_on_lang=False).

    On-task (MT-CL):   eval_dataset == fine_tuned_dataset
                       and eval_language != fine_tuned_language
    Off-task:          eval_dataset != fine_tuned_dataset

    harm_threshold is a positive magnitude: harm if Δ <= -harm_threshold.

    Returns a DataFrame with one row per task and columns:
      ['task', 'on_delta', 'on_positive_transfer_rate_pct', 'on_harm_pct',
       'off_delta', 'off_positive_transfer_rate_pct', 'off_harm_pct'].
    """
    required = {task_col, ft_lang_col, eval_task_col, eval_lang_col, delta_col}
    _check_required(df, required, df_name="df")

    dfx = _prepare_delta(df, delta_col)

    rows = []
    for task, g in dfx.groupby(task_col):
        # On-task: same dataset, different language (MT–CL)
        on_mask = (g[eval_task_col] == task) & (g[eval_lang_col] != g[ft_lang_col])

        # Off-task: all other datasets
        off_mask = g[eval_task_col] != task

        def agg(mask):
            d = g.loc[mask, "delta"]
            if d.empty:
                return np.nan, np.nan, np.nan, 0
            mean_delta = float(d.mean())
            positive_transfer_rate_pct = float((d > 0).mean() * 100.0)
            harm_pct = float((d <= -float(harm_threshold)).mean() * 100.0)
            return mean_delta, positive_transfer_rate_pct, harm_pct, int(d.size)

        on_delta, on_positive, on_harm, on_n = agg(on_mask)
        off_delta, off_positive, off_harm, off_n = agg(off_mask)

        rows.append(
            dict(
                task=task,
                on_delta=on_delta,
                on_positive_transfer_rate_pct=on_positive,
                on_harm_pct=on_harm,
                off_delta=off_delta,
                off_positive_transfer_rate_pct=off_positive,
                off_harm_pct=off_harm,
            )
        )

    summary = pd.DataFrame(rows).sort_values("task").reset_index(drop=True)
    return summary
