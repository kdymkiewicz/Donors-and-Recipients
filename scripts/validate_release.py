"""Fail-fast validation for the public camera-ready artifact."""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = {
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Llama-3.1-8B-Instruct",
    "Qwen2.5-0.5B-Instruct",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "gemma-3-1b-it",
    "gemma-3-4b-it",
}
DATASETS = {"arc_challenge", "global_mmlu", "hellaswag", "truthfulqa"}
LANGUAGES = {"ar", "bn", "de", "en", "es", "fr", "hi", "id", "it", "pt", "zh"}
REGIME_COUNTS = {"MT-ML": 396, "MT-CL": 3960, "CT-ML": 1188, "CT-CL": 11880}
MANIFEST_COUNTS = {
    "arc_challenge": {"train": 300, "test": 400},
    "global_mmlu": {"train": 215, "test": 400},
    "hellaswag": {"train": 300, "test": 400},
    "truthfulqa": {"train": 300, "test": 400},
}
KEY_COLUMNS = [
    "model_name",
    "fine_tuned_dataset",
    "fine_tuned_language",
    "eval_dataset",
    "eval_language",
]
NUMERIC_COLUMNS = [
    "model_size",
    "score_mean",
    "score_std",
    "score_base_mean",
    "score_base_std",
    "score_norm_mean",
    "score_norm_std",
    "score_norm_base_mean",
    "score_norm_base_std",
    "delta_score_mean",
    "delta_score_std",
    "delta_score_norm_mean",
    "delta_score_norm_std",
    "n_seeds",
    "score_se",
    "score_base_se",
    "score_norm_se",
    "score_norm_base_se",
    "delta_score_se",
    "delta_score_norm_se",
]


def fail(message: str) -> None:
    raise ValueError(message)


def derive_regime(frame: pd.DataFrame) -> pd.Series:
    same_task = frame["fine_tuned_dataset"].eq(frame["eval_dataset"])
    same_language = frame["fine_tuned_language"].eq(frame["eval_language"])
    return (same_task.astype(int) * 2 + same_language.astype(int)).map(
        {3: "MT-ML", 2: "MT-CL", 1: "CT-ML", 0: "CT-CL"}
    )


def assert_close(label: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
        fail(f"{label}: expected {expected:.12f}, found {actual:.12f}")


def validate_results(path: Path) -> None:
    frame = pd.read_csv(path)
    required = set(KEY_COLUMNS + NUMERIC_COLUMNS + ["regime"])
    missing = sorted(required - set(frame.columns))
    if missing:
        fail(f"{path}: missing columns: {missing}")
    if len(frame) != 17_424:
        fail(f"{path}: expected 17,424 rows, found {len(frame):,}")
    if frame.duplicated(KEY_COLUMNS).any():
        fail(f"{path}: duplicate ordered cell keys")
    if set(frame["model_name"]) != MODELS:
        fail(f"{path}: unexpected model set")
    for prefix in ("fine_tuned", "eval"):
        if set(frame[f"{prefix}_dataset"]) != DATASETS:
            fail(f"{path}: unexpected {prefix} dataset set")
        if set(frame[f"{prefix}_language"]) != LANGUAGES:
            fail(f"{path}: unexpected {prefix} language set")
    if not (frame["n_seeds"] == 3).all():
        fail(f"{path}: every row must aggregate exactly three seeds")
    numeric = frame[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy()).all():
        fail(f"{path}: numeric columns contain missing or non-finite values")
    expected_regime = derive_regime(frame)
    if not frame["regime"].equals(expected_regime):
        fail(f"{path}: regime labels do not match source/target keys")
    if frame["regime"].value_counts().to_dict() != REGIME_COUNTS:
        fail(f"{path}: incorrect regime counts")

    transfer = frame[frame["regime"] != "MT-ML"]
    delta = transfer["delta_score_norm_mean"]
    assert_close("transfer mean", float(delta.mean()), 0.8935821392216741)
    assert_close("transfer median", float(delta.median()), 0.2499999999999983)
    assert_close(
        "transfer positive-transfer rate",
        float((delta > 0).mean() * 100),
        59.67817712003759,
    )
    assert_close(
        "transfer harm rate",
        float((delta <= -1).mean() * 100),
        9.085036410617805,
    )

    expected = {
        "MT-ML": (2.0202020202020203, 68.18181818181817, 9.343434343434344),
        "MT-CL": (1.2464646464646463, 66.41414141414141, 7.121212121212121),
        "CT-ML": (0.8077300785634120, 56.90235690235690, 11.363636363636363),
        "CT-CL": (0.7845398428731761, 57.71043771043771, 9.511784511784512),
    }
    for regime, (mean, positive_rate, harm) in expected.items():
        values = frame.loc[frame["regime"] == regime, "delta_score_norm_mean"]
        assert_close(f"{regime} mean", float(values.mean()), mean)
        assert_close(
            f"{regime} positive-transfer rate",
            float((values > 0).mean() * 100),
            positive_rate,
        )
        assert_close(f"{regime} harm rate", float((values <= -1).mean() * 100), harm)
    print("OK results: 17,424 unique cells; exact grid, regimes, seeds, and headline metrics")


def validate_manifests(directory: Path) -> None:
    for benchmark, expected_counts in MANIFEST_COUNTS.items():
        path = directory / f"{benchmark}.csv"
        ids = {"train": set(), "test": set()}
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["split", "id"]:
                fail(f"{path}: expected columns split,id")
            for row in reader:
                split, identifier = row["split"], row["id"]
                if split not in ids:
                    fail(f"{path}: invalid split {split!r}")
                if identifier in ids[split]:
                    fail(f"{path}: duplicate {split} ID {identifier!r}")
                ids[split].add(identifier)
        counts = {split: len(values) for split, values in ids.items()}
        if counts != expected_counts:
            fail(f"{path}: expected {expected_counts}, found {counts}")
        if ids["train"] & ids["test"]:
            fail(f"{path}: train/test ID overlap")
    print("OK manifests: expected sizes, unique IDs, and disjoint train/test sets")


def validate_release_contents(root: Path) -> None:
    required = [
        "LICENSE",
        "NOTICE",
        "DATASETS.md",
        "requirements-analysis.txt",
        "requirements-training.txt",
        "results/LICENSE",
        "results/README.md",
        "results/transfer_results.csv",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        fail(f"Missing release files: {missing}")
    if "Apache License" not in (root / "LICENSE").read_text(errors="replace"):
        fail("Root LICENSE is not Apache 2.0 text")
    cc_text = (root / "results/LICENSE").read_text(errors="replace")
    if "Attribution 4.0 International" not in cc_text:
        fail("results/LICENSE is not CC BY 4.0 legal text")

    result_files = {
        path.name for path in (root / "results").iterdir() if path.is_file()
    }
    if result_files != {"LICENSE", "README.md", "transfer_results.csv"}:
        fail(f"Unexpected result artifacts: {sorted(result_files)}")

    excluded_dirs = {".git", ".idea", ".venv", "venv", "data", "outputs", "output", "__pycache__"}
    weight_suffixes = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}
    forbidden_text = {
        "private path": re.compile(r"/(?:Users|home|scratch)/[^\s'\"]+"),
        "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
        "API key assignment": re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token)\s*[=:]\s*['\"][^'\"]+"),
        "obsolete regime label": re.compile(r"\bML-CT\b"),
        "obsolete config reference": re.compile(r"\bconfig\.yaml\b"),
        "cluster placeholder": re.compile(r"\byour_(?:checkpoint|adapter|merged|base)_"),
    }
    violations = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in excluded_dirs for part in relative.parts):
            continue
        if path.is_dir():
            if path.name in {"wandb", ".wandb"}:
                violations.append(f"W&B metadata directory: {relative}")
            continue
        if path.suffix.lower() in weight_suffixes:
            violations.append(f"model-weight-like file: {relative}")
        if path == root / "results" / "transfer_results.csv" or path.suffix == ".csv":
            continue
        if path.stat().st_size > 2_000_000:
            continue
        text = path.read_text(errors="replace")
        for label, pattern in forbidden_text.items():
            if pattern.search(text):
                violations.append(f"{label}: {relative}")
    if violations:
        fail("Release content scan failed:\n  " + "\n  ".join(sorted(set(violations))))
    print("OK contents: licences present; no ablation data, credentials, private paths, weights, or obsolete labels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the camera-ready release artifact.")
    parser.add_argument("--results", type=Path, default=Path("results/transfer_results.csv"))
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("split_manifests/canonical")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        validate_results(args.results)
        validate_manifests(args.manifest_dir)
        validate_release_contents(root)
    except (FileNotFoundError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    print("Release validation passed.")


if __name__ == "__main__":
    main()
