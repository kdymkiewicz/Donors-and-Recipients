"""Validate every public aligned dataset config against the canonical manifests."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from datasets import load_dataset


LANGUAGES = ("ar", "bn", "de", "en", "es", "fr", "hi", "id", "it", "pt", "zh")
DATASETS = {
    "arc_challenge": (
        "Dr4kl3s/arc_challenge_core_grid_seed42",
        "id",
    ),
    "global_mmlu": (
        "Dr4kl3s/global_mmlu_lite_core_grid_seed42",
        "sample_id",
    ),
    "hellaswag": (
        "Dr4kl3s/hellaswag_coregrid_seed42",
        "id",
    ),
    "truthfulqa": (
        "Dr4kl3s/truthfulqa_coregrid_seed42",
        "truth_id",
    ),
}


def read_manifest(path: Path) -> dict[str, set[str]]:
    expected = {"train": set(), "test": set()}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["split", "id"]:
            raise ValueError(f"{path}: expected columns ['split', 'id']")
        for row in reader:
            split = row["split"]
            if split not in expected:
                raise ValueError(f"{path}: invalid split {split!r}")
            identifier = row["id"]
            if identifier in expected[split]:
                raise ValueError(f"{path}: duplicate {split} ID {identifier!r}")
            expected[split].add(identifier)
    if expected["train"] & expected["test"]:
        raise ValueError(f"{path}: train/test overlap")
    return expected


def validate_config(
    *,
    benchmark: str,
    repo_id: str,
    language: str,
    id_column: str,
    manifest: dict[str, set[str]],
    revision: str,
) -> None:
    dataset = load_dataset(repo_id, language, revision=revision)
    if set(dataset) != {"train", "test"}:
        raise ValueError(
            f"{repo_id}/{language}: expected train and test splits, got {sorted(dataset)}"
        )
    for split in ("train", "test"):
        if id_column not in dataset[split].column_names:
            raise ValueError(f"{repo_id}/{language}/{split}: missing {id_column!r}")
        ids = [str(value) for value in dataset[split][id_column]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{repo_id}/{language}/{split}: duplicate IDs")
        got = set(ids)
        if got != manifest[split]:
            raise ValueError(
                f"{repo_id}/{language}/{split}: manifest mismatch "
                f"(missing={len(manifest[split] - got)}, extra={len(got - manifest[split])})"
            )
    print(f"OK {benchmark}/{language}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate all public configs against canonical split manifests."
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("split_manifests/canonical"),
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub revision to validate (default: main).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for benchmark, (repo_id, id_column) in DATASETS.items():
        manifest = read_manifest(args.manifest_dir / f"{benchmark}.csv")
        for language in LANGUAGES:
            validate_config(
                benchmark=benchmark,
                repo_id=repo_id,
                language=language,
                id_column=id_column,
                manifest=manifest,
                revision=args.revision,
            )
    print("All 44 public dataset configurations match the canonical manifests.")


if __name__ == "__main__":
    main()
