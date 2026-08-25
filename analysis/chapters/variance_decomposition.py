# analysis/variance_decomposition.py

from typing import Dict, Optional

import numpy as np
import pandas as pd
import warnings
from scipy.sparse import SparseEfficiencyWarning
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", SparseEfficiencyWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


REGIME_ORDER = ["MT-CL", "CT-ML", "CT-CL"]
COMPONENT_ORDER = ["Model", "Source", "Target", "Residual"]


def _add_regime_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'regime' column:

      MT-CL : same dataset, different language
      CT-ML : different dataset, same language
      CT-CL : different dataset AND different language

    Rows with same dataset & same language get regime = None.
    """
    df = df.copy()

    same_task = df["fine_tuned_dataset"].astype(str) == df["eval_dataset"].astype(str)
    same_lang = df["fine_tuned_language"].astype(str) == df["eval_language"].astype(str)

    df["regime"] = np.select(
        [
            same_task & ~same_lang,  # matched task, cross-language
            ~same_task & same_lang,  # cross-task, matched language
            ~same_task & ~same_lang,  # cross-task, cross-language
        ],
        ["MT-CL", "CT-ML", "CT-CL"],
        default=None,
    )

    return df


def _pick_model_column(df: pd.DataFrame) -> str:
    """Use 'model_name' if present, otherwise fall back to 'model'."""
    if "model_name" in df.columns:
        return "model_name"
    if "model" in df.columns:
        return "model"
    raise KeyError("Expected a 'model_name' or 'model' column in df_agg.")


def _fit_mixedlm_variance(
    df: pd.DataFrame,
    *,
    delta_col: str,
    quiet: bool = True,
) -> Dict[str, float]:
    """
    Fit a simple mixed-effects model:

        delta ~ 1

    Random components:
      - random intercept for model_id        -> Model
      - VC for source_id (FT dataset+lang)  -> Source
      - VC for target_id (eval dataset+lang)-> Target

    Returns variance shares (%) for:
      Model, Source, Target, Residual
    """
    work = df.copy()

    if delta_col not in work.columns:
        raise KeyError(f"delta_col '{delta_col}' not found in DataFrame.")

    work[delta_col] = pd.to_numeric(work[delta_col], errors="coerce")
    work = work.dropna(subset=[delta_col])
    if work.empty:
        raise ValueError("No valid rows after coercing delta_col to numeric.")

    # required columns
    required = [
        "fine_tuned_dataset",
        "fine_tuned_language",
        "eval_dataset",
        "eval_language",
    ]
    missing = [c for c in required if c not in work.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    model_col = _pick_model_column(work)

    # IDs for random components
    work["model_id"] = work[model_col].astype(str)
    work["source_id"] = (
        work["fine_tuned_dataset"].astype(str)
        + "⟂"
        + work["fine_tuned_language"].astype(str)
    )
    work["target_id"] = (
        work["eval_dataset"].astype(str) + "⟂" + work["eval_language"].astype(str)
    )

    # cast to category for efficiency / stability
    for col in ["model_id", "source_id", "target_id"]:
        work[col] = work[col].astype("category")

    # statsmodels formula uses 'delta' as the response name
    work = work.rename(columns={delta_col: "delta"})

    # MixedLM: random intercept per model_id, plus VC for source/target
    md = MixedLM.from_formula(
        formula="delta ~ 1",
        groups="model_id",
        re_formula="1",
        vc_formula={
            "source": "0 + C(source_id)",
            "target": "0 + C(target_id)",
        },
        data=work,
        use_sparse=True,
    )

    result = md.fit(reml=True, method="lbfgs", maxiter=1000, disp=not quiet)

    components: Dict[str, float] = {}

    # Model component: random intercept variance
    try:
        model_var = float(result.cov_re.iloc[0, 0])
        if np.isfinite(model_var) and model_var >= 0:
            components["Model"] = model_var
    except Exception:
        if not quiet:
            print("Warning: could not extract model (random intercept) variance.")

    # Source / Target components: vcomp follows vc_formula order
    vcomp = np.asarray(getattr(result, "vcomp", []), dtype=float).ravel()
    vc_names = ["Source", "Target"]
    for name, var in zip(vc_names, vcomp):
        if np.isfinite(var) and var >= 0:
            components[name] = float(var)

    # Residual
    resid_var = float(result.scale)
    if np.isfinite(resid_var) and resid_var >= 0:
        components["Residual"] = resid_var

    # Convert to percentage shares
    total = sum(components.values())
    if total <= 0:
        shares = {k: np.nan for k in COMPONENT_ORDER}
    else:
        shares = {
            comp: components.get(comp, 0.0) / total * 100.0 for comp in COMPONENT_ORDER
        }

    return shares


def variance_decomposition(
    df_agg: pd.DataFrame,
    *,
    delta_col: str = "delta_score_norm_mean",
    min_rows: int = 50,
    quiet: bool = True,
) -> pd.DataFrame:
    """
    Run variance decomposition of Δ by model / source / target.

    - Fits one mixed model over all rows ("Overall").
    - Fits separate models within each transfer regime:
        MT-CL (matched task, cross-language)
        CT-ML (cross-task, matched language)
        CT-CL (cross-task, cross-language)
      skipping regimes with fewer than `min_rows` rows.

    Returns a DataFrame with components as rows and
    ["Overall", "MT-CL", "CT-ML", "CT-CL"] as columns (where available),
    and also prints it via `print_df`.
    """
    df = _add_regime_column(df_agg)

    results: Dict[str, Dict[str, float]] = {}

    # Overall
    overall_shares = _fit_mixedlm_variance(df, delta_col=delta_col, quiet=quiet)
    results["Overall"] = overall_shares

    # Per-regime
    for regime in REGIME_ORDER:
        sub = df[df["regime"] == regime]
        if len(sub) < min_rows:
            if not quiet:
                print(f"[{regime}] skipped (n={len(sub)} < {min_rows})")
            continue

        shares = _fit_mixedlm_variance(sub, delta_col=delta_col, quiet=quiet)
        results[regime] = shares

    # Assemble table: components as rows, regimes as columns
    var_table = pd.DataFrame.from_dict(results, orient="columns").reindex(
        index=COMPONENT_ORDER
    )

    return var_table
