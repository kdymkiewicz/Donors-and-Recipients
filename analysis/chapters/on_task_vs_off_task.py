import numpy as np
import pandas as pd


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
    Per-task on-task vs off-task statistics, consistent with pareto._compute_on_off
    when strict_on_lang=False.

    On-task (MT-CL):   eval_dataset == fine_tuned_dataset
                       and eval_language != fine_tuned_language
    Off-task:          eval_dataset != fine_tuned_dataset

    harm_threshold is a positive magnitude: harm if Δ <= -harm_threshold.

    Returns a DataFrame with one row per task and columns:
      ['task', 'on_n', 'on_delta', 'on_positive_transfer_rate_pct', 'on_harm_pct',
       'off_n', 'off_delta', 'off_positive_transfer_rate_pct', 'off_harm_pct'].
    """

    required = [task_col, ft_lang_col, eval_task_col, eval_lang_col, delta_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in df: {missing}")

    # Ensure delta is numeric like in pareto._compute_on_off
    df = df.copy()
    df["__delta__"] = pd.to_numeric(df[delta_col], errors="coerce")

    rows = []
    for task, g in df.groupby(task_col):
        # On-task: same dataset, different language (MT–CL)
        on_mask = (g[eval_task_col] == task) & (g[eval_lang_col] != g[ft_lang_col])

        # Off-task: all other datasets
        off_mask = g[eval_task_col] != task

        def agg(mask):
            d = g.loc[mask, "__delta__"]
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
                on_n=on_n,
                on_delta=on_delta,
                on_positive_transfer_rate_pct=on_positive,
                on_harm_pct=on_harm,
                off_n=off_n,
                off_delta=off_delta,
                off_positive_transfer_rate_pct=off_positive,
                off_harm_pct=off_harm,
            )
        )

    summary = pd.DataFrame(rows).sort_values("task").reset_index(drop=True)
    return summary
