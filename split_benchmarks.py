"""Materialise the four language-aligned paper datasets from frozen ID manifests.

Local output is the default. Hugging Face publication occurs only when the
caller supplies both ``--publish`` and ``--hf-user``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


LANGUAGES = ("ar", "bn", "de", "en", "es", "fr", "hi", "id", "it", "pt", "zh")
BENCHMARKS = ("arc_challenge", "global_mmlu", "hellaswag", "truthfulqa")


@dataclass(frozen=True)
class SourceSpec:
    dataset: str
    revision: str


SOURCES = {
    "arc_challenge": SourceSpec(
        "alexandrainst/m_arc", "69b0991ee606079105c4f9e271bcc1fef0014e03"
    ),
    "global_mmlu": SourceSpec(
        "CohereLabs/Global-MMLU-Lite",
        "36c2fd756f19ccf13a9a96c8e53ccecc02192b8b",
    ),
    "hellaswag_main": SourceSpec(
        "alexandrainst/m_hellaswag",
        "9d31dc982bd6285e081e3e3136332a38b9c1d7b7",
    ),
    "hellaswag_zh": SourceSpec(
        "richmondsin/m_hellaswag",
        "8d9642f16ae3031f7eddc3e3531b6ae58686d81a",
    ),
    "truthfulqa_main": SourceSpec(
        "alexandrainst/m_truthfulqa",
        "f0445d470f1925882b990f5f247fdcf288972f60",
    ),
    "truthfulqa_en": SourceSpec(
        "Dr4kl3s/truthfulqa_en_aligned",
        "7332da38d96e69b5d6f2502cd45db3fe011b2f19",
    ),
}

ID_COLUMNS = {
    "arc_challenge": "id",
    "global_mmlu": "sample_id",
    "hellaswag": "id",
    "truthfulqa": "truth_id",
}

HUB_REPO_NAMES = {
    "arc_challenge": "arc_challenge_core_grid_seed42",
    "global_mmlu": "global_mmlu_lite_core_grid_seed42",
    "hellaswag": "hellaswag_coregrid_seed42",
    "truthfulqa": "truthfulqa_coregrid_seed42",
}

NP_ARRAY_RE = re.compile(
    r"array\((\[.*?\])\s*,\s*dtype=[^)]*\)",
    re.DOTALL,
)


def concat_splits(ds_dict: DatasetDict | dict[str, Dataset], splits: Iterable[str]) -> Dataset:
    parts = [ds_dict[split] for split in splits]
    if not parts:
        raise ValueError("At least one source split is required")
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def read_manifest(path: Path) -> dict[str, set[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    ids: dict[str, set[str]] = {"train": set(), "test": set()}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["split", "id"]:
            raise ValueError(f"{path}: expected columns ['split', 'id']")
        for row in reader:
            split = row["split"]
            identifier = row["id"]
            if split not in ids:
                raise ValueError(f"{path}: invalid split {split!r}")
            if identifier in ids[split]:
                raise ValueError(f"{path}: duplicate {split} ID {identifier!r}")
            ids[split].add(identifier)
    overlap = ids["train"] & ids["test"]
    if overlap:
        raise ValueError(f"{path}: train/test IDs overlap: {sorted(overlap)[:5]}")
    return ids


def filter_by_ids(dataset: Dataset, id_column: str, ids: set[str]) -> Dataset:
    available = {str(value) for value in dataset[id_column]}
    missing = ids - available
    if missing:
        raise ValueError(
            f"Source pool lacks {len(missing)} manifest IDs for {id_column}: "
            f"{sorted(missing)[:5]}"
        )
    selected = dataset.filter(
        lambda example, keep=ids, column=id_column: str(example[column]) in keep
    )
    return selected.sort(id_column)


def validate_materialised(
    dataset: DatasetDict,
    *,
    benchmark: str,
    language: str,
    manifest: dict[str, set[str]],
) -> None:
    id_column = ID_COLUMNS[benchmark]
    if set(dataset) != {"train", "test"}:
        raise ValueError(f"{benchmark}/{language}: expected train and test splits")
    for split in ("train", "test"):
        got = [str(value) for value in dataset[split][id_column]]
        if len(got) != len(set(got)):
            raise ValueError(f"{benchmark}/{language}/{split}: duplicate IDs")
        if set(got) != manifest[split]:
            raise ValueError(f"{benchmark}/{language}/{split}: manifest mismatch")
    if set(map(str, dataset["train"][id_column])) & set(
        map(str, dataset["test"][id_column])
    ):
        raise ValueError(f"{benchmark}/{language}: train/test overlap")


def add_synthetic_id(
    ds_dict: DatasetDict | dict[str, Dataset], id_column: str
) -> DatasetDict:
    """Assign stable row-order IDs, unique across the source splits."""
    output = {}
    offset = 0
    for split_name, dataset in ds_dict.items():
        output[split_name] = dataset.map(
            lambda _example, index, base=offset: {id_column: base + index},
            with_indices=True,
        )
        offset += len(dataset)
    return DatasetDict(output)


def parse_mc_target(value: object) -> dict[str, list]:
    if isinstance(value, dict):
        parsed = value
    else:
        clean = NP_ARRAY_RE.sub(r"\1", str(value).strip())
        parsed = ast.literal_eval(clean)
    return {
        "choices": [str(choice) for choice in parsed["choices"]],
        "labels": [int(label) for label in parsed["labels"]],
    }


def normalise_truthfulqa_en_schema(ds_dict: DatasetDict) -> DatasetDict:
    def convert(example: dict) -> dict:
        mc1 = parse_mc_target(example["mc1_targets"])
        mc2 = parse_mc_target(example["mc2_targets"])
        return {
            "mc1_targets_choices": mc1["choices"],
            "mc1_targets_labels": mc1["labels"],
            "mc2_targets_choices": mc2["choices"],
            "mc2_targets_labels": mc2["labels"],
        }

    output = {}
    for split_name, dataset in ds_dict.items():
        converted = dataset.map(convert)
        old_columns = [
            column
            for column in ("mc1_targets", "mc2_targets")
            if column in converted.column_names
        ]
        output[split_name] = converted.remove_columns(old_columns)
    return DatasetDict(output)


def load_standard_pool(benchmark: str, language: str) -> tuple[Dataset, Dataset]:
    spec = SOURCES[benchmark]
    dataset = load_dataset(spec.dataset, language, revision=spec.revision)
    if benchmark == "arc_challenge":
        return concat_splits(dataset, ["train"]), concat_splits(dataset, ["test"])
    if benchmark == "global_mmlu":
        return concat_splits(dataset, ["dev"]), concat_splits(dataset, ["test"])
    raise ValueError(f"Unsupported standard benchmark: {benchmark}")


def load_hellaswag_pool(language: str) -> tuple[Dataset, Dataset]:
    spec = SOURCES["hellaswag_zh" if language == "zh" else "hellaswag_main"]
    dataset = load_dataset(spec.dataset, language, split="val", revision=spec.revision)
    return dataset, dataset


def load_truthfulqa_pool(language: str) -> tuple[Dataset, Dataset]:
    if language == "en":
        spec = SOURCES["truthfulqa_en"]
        dataset = load_dataset(spec.dataset, revision=spec.revision)
        dataset = normalise_truthfulqa_en_schema(dataset)
        dataset = add_synthetic_id(dataset, ID_COLUMNS["truthfulqa"])
        pool = concat_splits(dataset, ["validation"])
    else:
        spec = SOURCES["truthfulqa_main"]
        dataset = load_dataset(spec.dataset, language, revision=spec.revision)
        dataset = add_synthetic_id(dataset, ID_COLUMNS["truthfulqa"])
        pool = concat_splits(dataset, ["val"])
    return pool, pool


LOADERS: dict[str, Callable[[str], tuple[Dataset, Dataset]]] = {
    "arc_challenge": lambda language: load_standard_pool("arc_challenge", language),
    "global_mmlu": lambda language: load_standard_pool("global_mmlu", language),
    "hellaswag": load_hellaswag_pool,
    "truthfulqa": load_truthfulqa_pool,
}


def emit_dataset(
    dataset: DatasetDict,
    *,
    benchmark: str,
    language: str,
    output_dir: Path,
    overwrite: bool,
    publish: bool,
    hf_user: str | None,
) -> None:
    destination = output_dir / benchmark / language
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"{destination} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(destination)
    print(f"saved {benchmark}/{language} -> {destination}")

    if publish:
        if not hf_user:
            raise ValueError("--hf-user is required with --publish")
        repo_id = f"{hf_user}/{HUB_REPO_NAMES[benchmark]}"
        dataset.push_to_hub(repo_id, config_name=language, private=False)
        print(f"published {benchmark}/{language} -> {repo_id}")


def materialise_benchmark(
    benchmark: str,
    *,
    languages: Iterable[str],
    manifest_dir: Path,
    output_dir: Path,
    overwrite: bool,
    publish: bool,
    hf_user: str | None,
) -> None:
    manifest = read_manifest(manifest_dir / f"{benchmark}.csv")
    id_column = ID_COLUMNS[benchmark]
    loader = LOADERS[benchmark]
    for language in languages:
        train_pool, test_pool = loader(language)
        dataset = DatasetDict(
            {
                "train": filter_by_ids(train_pool, id_column, manifest["train"]),
                "test": filter_by_ids(test_pool, id_column, manifest["test"]),
            }
        )
        validate_materialised(
            dataset,
            benchmark=benchmark,
            language=language,
            manifest=manifest,
        )
        emit_dataset(
            dataset,
            benchmark=benchmark,
            language=language,
            output_dir=output_dir,
            overwrite=overwrite,
            publish=publish,
            hf_user=hf_user,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialise the four paper datasets locally from checked-in canonical "
            "ID manifests. Publication is opt-in."
        )
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("split_manifests/canonical"),
        help="Directory containing one <benchmark>.csv ID manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/aligned"),
        help="Local destination for DatasetDict objects.",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=BENCHMARKS,
        default=list(BENCHMARKS),
    )
    parser.add_argument(
        "--languages", nargs="+", choices=LANGUAGES, default=list(LANGUAGES)
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing local benchmark/language directory.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Explicitly publish each materialised language config to Hugging Face.",
    )
    parser.add_argument(
        "--hf-user",
        help="Hugging Face account used only with --publish.",
    )
    args = parser.parse_args()
    if args.publish and not args.hf_user:
        parser.error("--hf-user is required with --publish")
    return args


def main() -> None:
    args = parse_args()
    print(f"split=canonical; publish={args.publish}")
    for benchmark in args.benchmarks:
        materialise_benchmark(
            benchmark,
            languages=args.languages,
            manifest_dir=args.manifest_dir,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            publish=args.publish,
            hf_user=args.hf_user,
        )


if __name__ == "__main__":
    main()
