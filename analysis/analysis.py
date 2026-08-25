"""Reproduce the paper-facing tables and figures from the released main grid."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "donors-matplotlib-cache")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "donors-xdg-cache")
)
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from chapters.consistency_index import (
    consistency_index,
    consistency_index_by_family,
    consistency_index_by_size,
)
from chapters.donor_recipient_roles import (
    compute_language_roles_by_family,
    compute_task_roles_from_agg,
    language_donor_correlations_mtcl,
    plot_language_donor_recipient,
)
from chapters.lang_lang_heatmap import plot_language_transfer_heatmap
from chapters.ling_correlation import analyze_linguistic_correlations
from chapters.pareto_plot import (
    compute_task_on_off_summary,
    plot_pareto_family_summary,
    plot_pareto_on_off,
)
from chapters.task_task_heatmap import plot_task_transfer_heatmap
from chapters.uplift_and_transfer import (
    crosslingual_vs_crosstask,
    harmful_mtcl_breakdown,
    mtcl_by_construction_type,
    uplift,
)
from chapters.variance_decomposition import variance_decomposition
from utils import print_df


DELTA_COLUMN = "delta_score_norm_mean"
REQUIRED_COLUMNS = {
    "model_name",
    "model_size",
    "fine_tuned_dataset",
    "fine_tuned_language",
    "eval_dataset",
    "eval_language",
    "regime",
    DELTA_COLUMN,
    "n_seeds",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce paper tables and figures from transfer_results.csv."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/transfer_results.csv"),
        help="Released 17,424-row main-grid CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis"),
        help="Destination for generated plots and CSV tables.",
    )
    parser.add_argument(
        "--skip-linguistic",
        action="store_true",
        help="Skip the lang2vec correlation table.",
    )
    parser.add_argument(
        "--skip-mixed-effects",
        action="store_true",
        help="Skip the slower mixed-effects variance decompositions.",
    )
    return parser.parse_args()


def load_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Results file not found: {path}")
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if len(frame) != 17_424:
        raise ValueError(f"Expected 17,424 rows, found {len(frame):,}")
    return frame


def save_table(name: str, frame: pd.DataFrame, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(frame.index, pd.RangeIndex):
        if isinstance(frame.index, pd.MultiIndex):
            names = [
                name or f"index_level_{position}"
                for position, name in enumerate(frame.index.names)
            ]
            frame = frame.rename_axis(names).reset_index()
        else:
            frame = frame.rename_axis(frame.index.name or "row").reset_index()
    path = tables_dir / f"{name}.csv"
    frame.to_csv(path, index=False)
    print_df(name.replace("_", " ").title(), frame, max_rows=20)


def run_analysis(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    skip_linguistic: bool,
    skip_mixed_effects: bool,
) -> None:
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    on_source = frame[frame["regime"] == "MT-ML"].copy()
    transfer = frame[frame["regime"] != "MT-ML"].copy()

    save_table("global_uplift", uplift(transfer, column=DELTA_COLUMN), tables_dir)
    save_table(
        "regime_summary",
        crosslingual_vs_crosstask(
            transfer,
            delta_col=DELTA_COLUMN,
            harm_threshold=1.0,
            df_on_task=on_source,
        ),
        tables_dir,
    )
    save_table(
        "mtcl_harm_breakdown",
        harmful_mtcl_breakdown(
            transfer,
            delta_col=DELTA_COLUMN,
            harm_threshold=1.0,
            size_billion_col="model_size",
            model_col="model_name",
        ),
        tables_dir,
    )
    save_table(
        "mtcl_by_construction_type",
        mtcl_by_construction_type(
            frame, delta_col=DELTA_COLUMN, harm_threshold=1.0
        ),
        tables_dir,
    )

    if not skip_linguistic:
        linguistic = analyze_linguistic_correlations(
            transfer, delta_col=DELTA_COLUMN, verbose=False
        )
        save_table("linguistic_correlations", linguistic, tables_dir)

    pareto = plot_pareto_on_off(
        transfer,
        delta_col=DELTA_COLUMN,
        out_path=plots_dir / "pareto_on_off.png",
        return_full=True,
        format_dataset_labels=False,
    )
    save_table("pareto_profiles", pareto, tables_dir)
    pareto_summary = plot_pareto_family_summary(
        transfer,
        delta_col=DELTA_COLUMN,
        out_path=plots_dir / "pareto_on_off_summary.png",
    )
    save_table("pareto_family_summary", pareto_summary, tables_dir)
    save_table(
        "task_on_off_summary",
        compute_task_on_off_summary(transfer, delta_col=DELTA_COLUMN),
        tables_dir,
    )

    language_roles = plot_language_donor_recipient(
        transfer,
        delta_col=DELTA_COLUMN,
        out_path=plots_dir / "language_donor_recipient.png",
    )
    save_table("language_donor_recipient", language_roles, tables_dir)
    save_table(
        "language_roles_by_family",
        compute_language_roles_by_family(transfer, delta_col=DELTA_COLUMN),
        tables_dir,
    )

    roles_all, roles_curated, roles_translated, rho = (
        language_donor_correlations_mtcl(
            frame,
            curated_dataset="global_mmlu",
            mt_datasets=["arc_challenge", "hellaswag", "truthfulqa"],
            delta_col=DELTA_COLUMN,
            min_rows_per_model=1,
        )
    )
    save_table("language_roles_all", roles_all, tables_dir)
    save_table("language_roles_curated", roles_curated, tables_dir)
    save_table("language_roles_translated", roles_translated, tables_dir)
    save_table(
        "curated_vs_translated_donor_correlation",
        pd.DataFrame([{"spearman_rho": rho}]),
        tables_dir,
    )
    save_table(
        "task_donor_recipient",
        compute_task_roles_from_agg(transfer, delta_col=DELTA_COLUMN),
        tables_dir,
    )

    task_matrix = plot_task_transfer_heatmap(
        transfer,
        delta_col=DELTA_COLUMN,
        out_path=plots_dir / "task_transfer_heatmap.png",
        annotate_threshold=1.0,
    )
    save_table(
        "task_transfer_matrix",
        task_matrix.rename_axis("source_task").reset_index(),
        tables_dir,
    )
    language_matrix = plot_language_transfer_heatmap(
        transfer,
        delta_col=DELTA_COLUMN,
        agg_fn="mean",
        out_path=plots_dir / "language_transfer_heatmap.png",
    )
    save_table(
        "language_transfer_matrix",
        language_matrix.rename_axis("source_language").reset_index(),
        tables_dir,
    )

    if not skip_mixed_effects:
        save_table(
            "variance_decomposition_all",
            variance_decomposition(transfer, delta_col=DELTA_COLUMN),
            tables_dir,
        )
        global_mmlu = transfer[
            (transfer["fine_tuned_dataset"] == "global_mmlu")
            | (transfer["eval_dataset"] == "global_mmlu")
        ]
        translated = transfer[
            (transfer["fine_tuned_dataset"] != "global_mmlu")
            & (transfer["eval_dataset"] != "global_mmlu")
        ]
        save_table(
            "variance_decomposition_global_mmlu",
            variance_decomposition(global_mmlu, delta_col=DELTA_COLUMN),
            tables_dir,
        )
        save_table(
            "variance_decomposition_translated",
            variance_decomposition(translated, delta_col=DELTA_COLUMN),
            tables_dir,
        )

    save_table(
        "consistency_by_model",
        consistency_index(transfer, delta_col=DELTA_COLUMN),
        tables_dir,
    )
    save_table(
        "consistency_by_family",
        consistency_index_by_family(transfer, delta_col=DELTA_COLUMN),
        tables_dir,
    )
    save_table(
        "consistency_by_size",
        consistency_index_by_size(transfer, delta_col=DELTA_COLUMN),
        tables_dir,
    )

    print(f"Analysis written to {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    frame = load_results(args.results)
    run_analysis(
        frame,
        args.output_dir,
        skip_linguistic=args.skip_linguistic,
        skip_mixed_effects=args.skip_mixed_effects,
    )


if __name__ == "__main__":
    main()
