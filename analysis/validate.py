from __future__ import annotations

import itertools
import os
import sys
from typing import Optional, Sequence, Tuple

import pandas as pd

EXPECTED_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-3-1b-it",
    "google/gemma-3-4b-it",
]
EXPECTED_DATASETS = ["arc_challenge", "global_mmlu", "truthfulqa", "hellaswag"]
EXPECTED_LANGS = ["ar", "bn", "de", "en", "es", "fr", "hi", "id", "it", "pt", "zh"]
EXPECTED_SEEDS = [17, 42, 101]

REQUIRED_COLS = [
    "model_name",
    "seed",
    "fine_tuned_dataset",
    "fine_tuned_language",
    "eval_dataset",
    "eval_language",
]

NA = {"", "na", "n/a", "nan", "none", "null"}
MAX_SHOW_MISSING = 100


def canon_model(s: str) -> str:
    s = str(s).strip()
    return s.split("/")[-1] if "/" in s else s


def norm_str(x) -> str:
    return str(x).strip()


def norm_na(x) -> Optional[str]:
    s = norm_str(x)
    return None if s.lower() in NA else s


def expected_eval_index(
    models: list[str], seeds: list[int], datasets: list[str], langs: list[str]
) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        itertools.product(models, seeds, datasets, langs),
        names=["model", "seed", "dataset", "lang"],
    )


def expected_full_ft_index(
    models: list[str],
    seeds: list[int],
    ft_datasets: list[str],
    ft_langs: list[str],
    eval_datasets: list[str],
    eval_langs: list[str],
) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        itertools.product(
            models, seeds, ft_datasets, ft_langs, eval_datasets, eval_langs
        ),
        names=["model", "seed", "ft_dataset", "ft_lang", "eval_dataset", "eval_lang"],
    )


def keys_index_eval(
    df: pd.DataFrame, model_col: str, ds_col: str, lang_col: str
) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(
        df[[model_col, "seed", ds_col, lang_col]].rename(
            columns={model_col: "model", ds_col: "dataset", lang_col: "lang"}
        )
    )


def keys_index_full_ft(df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(
        df[
            [
                "model",
                "seed",
                "ft_dataset",
                "ft_language",
                "eval_dataset",
                "eval_language",
            ]
        ].rename(columns={"ft_language": "ft_lang", "eval_language": "eval_lang"})
    )


def dupes(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    c = df.groupby(key_cols, dropna=False).size().reset_index(name="n")
    return c[c["n"] > 1].sort_values("n", ascending=False)


def show_missing(
    title: str, missing_df: pd.DataFrame, sort_cols: Sequence[str]
) -> None:
    if missing_df.empty:
        return
    m = missing_df.sort_values(list(sort_cols))
    print(f"\n{title}: {len(m):,} missing", file=sys.stderr)
    print(m.head(MAX_SHOW_MISSING).to_string(index=False), file=sys.stderr)
    if len(m) > MAX_SHOW_MISSING:
        print(
            f"... (showing first {MAX_SHOW_MISSING}; save missing_df to CSV if needed)",
            file=sys.stderr,
        )


def validate_csv(path: str, expected_seed: int) -> Tuple[bool, int]:
    if not os.path.exists(path):
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return False, 1

    df = pd.read_csv(path)

    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        print(f"ERROR: {path}: missing columns: {missing_cols}", file=sys.stderr)
        return False, 1

    # Normalize
    df = df.copy()
    df["model"] = df["model_name"].map(canon_model)
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    df["eval_dataset"] = df["eval_dataset"].map(norm_str)
    df["eval_language"] = df["eval_language"].map(norm_str)
    df["ft_dataset"] = df["fine_tuned_dataset"].map(norm_na)
    df["ft_language"] = df["fine_tuned_language"].map(norm_na)

    if df["seed"].isna().any():
        bad = df[df["seed"].isna()][["model_name", "seed"]].head(10)
        print(
            f"ERROR: {path}: non-integer seed found (showing up to 10 rows):",
            file=sys.stderr,
        )
        print(bad.to_string(index=False), file=sys.stderr)
        return False, 1

    seeds_in_file = sorted(df["seed"].unique().tolist())
    if seeds_in_file != [expected_seed]:
        print(
            f"ERROR: {path}: expected exactly seed {expected_seed}, but found seeds {seeds_in_file}",
            file=sys.stderr,
        )
        return False, 1

    seeds = [expected_seed]
    exp_models = sorted({canon_model(m) for m in EXPECTED_MODELS})
    exp_eval = expected_eval_index(exp_models, seeds, EXPECTED_DATASETS, EXPECTED_LANGS)

    # Sanity check
    half_na = df["ft_dataset"].isna() ^ df["ft_language"].isna()
    if half_na.any():
        bad = df.loc[half_na, REQUIRED_COLS].head(20)
        print(
            f"ERROR: {path}: {int(half_na.sum())} rows have only one of fine_tuned_dataset/language as n/a.",
            file=sys.stderr,
        )
        print(bad.to_string(index=False), file=sys.stderr)
        return False, 1

    base = df[df["ft_dataset"].isna() & df["ft_language"].isna()]
    base_dupes = dupes(base, ["model", "seed", "eval_dataset", "eval_language"])
    base_idx = keys_index_eval(base, "model", "eval_dataset", "eval_language")
    base_missing = exp_eval.difference(base_idx).to_frame(index=False)

    ft = df[~(df["ft_dataset"].isna() & df["ft_language"].isna())]
    ft_matched = ft[
        (ft["ft_dataset"] == ft["eval_dataset"])
        & (ft["ft_language"] == ft["eval_language"])
    ]
    ft_dupes = dupes(ft_matched, ["model", "seed", "eval_dataset", "eval_language"])
    ft_idx = keys_index_eval(ft_matched, "model", "eval_dataset", "eval_language")
    ft_missing = exp_eval.difference(ft_idx).to_frame(index=False)

    exp_full_ft = expected_full_ft_index(
        exp_models,
        seeds,
        EXPECTED_DATASETS,
        EXPECTED_LANGS,
        EXPECTED_DATASETS,
        EXPECTED_LANGS,
    )
    ft_full_dupes = dupes(
        ft,
        ["model", "seed", "ft_dataset", "ft_language", "eval_dataset", "eval_language"],
    )
    ft_full_idx = keys_index_full_ft(ft)
    ft_full_missing = exp_full_ft.difference(ft_full_idx).to_frame(index=False)

    unexpected_models = sorted(set(df["model"]) - set(exp_models))
    unexpected_eval_ds = sorted(set(df["eval_dataset"]) - set(EXPECTED_DATASETS))
    unexpected_eval_lang = sorted(set(df["eval_language"]) - set(EXPECTED_LANGS))
    unexpected_ft_ds = sorted(set(ft["ft_dataset"].dropna()) - set(EXPECTED_DATASETS))
    unexpected_ft_lang = sorted(set(ft["ft_language"].dropna()) - set(EXPECTED_LANGS))

    valid = (
        base_missing.empty
        and ft_missing.empty
        and ft_full_missing.empty
        and base_dupes.empty
        and ft_dupes.empty
        and ft_full_dupes.empty
        and not unexpected_models
        and not unexpected_eval_ds
        and not unexpected_eval_lang
        and not unexpected_ft_ds
        and not unexpected_ft_lang
    )

    # Summary
    print(f"\n=== {os.path.basename(path)} ===")
    print(f"Seed: {expected_seed}")
    print(
        f"Expected per seed (eval grid): {len(exp_models)} models × {len(EXPECTED_DATASETS)} datasets × {len(EXPECTED_LANGS)} langs"
    )
    print(
        f"Base:        rows={len(base):,}        missing={len(base_missing):,}  dupes={len(base_dupes):,}"
    )
    print(
        f"FT matched:  rows={len(ft_matched):,}  missing={len(ft_missing):,}    dupes={len(ft_dupes):,}"
    )
    print(
        f"FT full:     rows={len(ft):,}          missing={len(ft_full_missing):,} dupes={len(ft_full_dupes):,}"
    )

    # Show exactly what's missing
    show_missing(
        "Missing BASE keys (model, seed, eval_dataset, eval_lang)",
        base_missing.rename(columns={"dataset": "eval_dataset", "lang": "eval_lang"}),
        sort_cols=["seed", "model", "eval_dataset", "eval_lang"],
    )
    show_missing(
        "Missing FINETUNED matched keys (model, seed, eval_dataset, eval_lang)",
        ft_missing.rename(columns={"dataset": "eval_dataset", "lang": "eval_lang"}),
        sort_cols=["seed", "model", "eval_dataset", "eval_lang"],
    )
    show_missing(
        "Missing FINETUNED full-grid keys (model, seed, ft_dataset, ft_lang, eval_dataset, eval_lang)",
        ft_full_missing,
        sort_cols=[
            "seed",
            "model",
            "ft_dataset",
            "ft_lang",
            "eval_dataset",
            "eval_lang",
        ],
    )

    # Dupes
    if not base_dupes.empty:
        print(f"\nDuplicate BASE keys: {len(base_dupes):,}", file=sys.stderr)
        print(base_dupes.to_string(index=False), file=sys.stderr)
    if not ft_dupes.empty:
        print(f"\nDuplicate FINETUNED matched keys: {len(ft_dupes):,}", file=sys.stderr)
        print(ft_dupes.to_string(index=False), file=sys.stderr)
    if not ft_full_dupes.empty:
        print(
            f"\nDuplicate FINETUNED full-grid keys: {len(ft_full_dupes):,}",
            file=sys.stderr,
        )
        print(ft_full_dupes.to_string(index=False), file=sys.stderr)

    # Unexpected values
    if (
        unexpected_models
        or unexpected_eval_ds
        or unexpected_eval_lang
        or unexpected_ft_ds
        or unexpected_ft_lang
    ):
        print("\nUnexpected values:", file=sys.stderr)
        if unexpected_models:
            print(f"  models: {unexpected_models}", file=sys.stderr)
        if unexpected_eval_ds:
            print(f"  eval_dataset: {unexpected_eval_ds}", file=sys.stderr)
        if unexpected_eval_lang:
            print(f"  eval_language: {unexpected_eval_lang}", file=sys.stderr)
        if unexpected_ft_ds:
            print(f"  fine_tuned_dataset: {unexpected_ft_ds}", file=sys.stderr)
        if unexpected_ft_lang:
            print(f"  fine_tuned_language: {unexpected_ft_lang}", file=sys.stderr)

    if valid:
        print("RESULT: VALID")
        return True, 0
    else:
        print("RESULT: INVALID", file=sys.stderr)
        return False, 1


def main() -> int:
    base_dir = "./output/data"
    files = [
        (os.path.join(base_dir, "results_seed17.csv"), 17),
        (os.path.join(base_dir, "results_seed42.csv"), 42),
        (os.path.join(base_dir, "results_seed101.csv"), 101),
    ]

    all_ok = True
    rc = 0
    for path, seed in files:
        ok, code = validate_csv(path, seed)
        all_ok = all_ok and ok
        rc = max(rc, code)

    if all_ok:
        print("\nOVERALL: ALL VALID")
        return 0
    else:
        print("\nOVERALL: SOME INVALID", file=sys.stderr)
        return rc


if __name__ == "__main__":
    raise SystemExit(main())
