import argparse
import csv
import json
import os
import re
from glob import glob
from typing import Dict, Any, List, Tuple, Optional


def extract_ft_info(
    configs: Dict[str, Any], eval_mode: str
) -> Tuple[Optional[str], str, str]:
    """
    Extract model_name and (optionally) fine-tuned dataset/language
    from configs.<task>.metadata.pretrained.

    eval_mode:
      - "base":    treat as base model (no fine-tuning info, use "n/a")
      - "adapter": require path like .../merged/<dataset>/<lang>/<model_name>
    """
    any_task = next(iter(configs.keys()))
    meta = configs[any_task].get("metadata", {})
    pretrained_path = meta.get("pretrained") or ""
    parts = pretrained_path.strip("/").split("/")

    if eval_mode == "base":
        model_name = parts[-1] if parts else None
        return model_name, "n/a", "n/a"

    if "merged" not in parts:
        raise ValueError(f"Expected 'merged' in adapter path, got: {pretrained_path}")

    idx = parts.index("merged")
    if len(parts) < idx + 4:
        raise ValueError(
            "Adapter path should look like .../merged/<dataset>/<language>/<model_name>, got: "
            f"{pretrained_path}"
        )

    ft_dataset = parts[idx + 1]
    ft_lang = parts[idx + 2]
    model_name = parts[idx + 3]
    return model_name, ft_dataset, ft_lang


def extract_model_size(model_name: str) -> str:
    m = re.search(r"(\d+(?:\.\d+)?)([bBkKmM])", model_name)
    if not m:
        raise ValueError(f"Unable to extract model size from model_name: {model_name}")
    num, _ = m.groups()
    return num


def extract_seed_from_configs(configs: Dict[str, Any]) -> str:
    any_task = next(iter(configs.keys()))
    ds_path = configs[any_task].get("dataset_path") or ""
    m = re.search(r"seed(\d+)", ds_path)
    if not m:
        raise ValueError(f"Unable to extract seed from dataset_path: {ds_path}")
    return m.group(1)


def get_eval_language(eval_dataset: str) -> str:
    parts = eval_dataset.split("_")

    if eval_dataset.startswith("global_mmlu_"):
        if len(parts) >= 3 and len(parts[2]) <= 3:
            return parts[2]
    elif eval_dataset.startswith("truthfulqa_"):
        if len(parts) >= 2 and len(parts[1]) <= 3:
            return parts[1]
    elif eval_dataset.startswith("arc_"):
        if len(parts) >= 2 and len(parts[1]) <= 3:
            return parts[1]
    elif eval_dataset.startswith("hellaswag_"):
        if len(parts) >= 2 and len(parts[1]) <= 3:
            return parts[1]

    raise ValueError(
        f"Unable to extract eval_language from eval_dataset: {eval_dataset}"
    )


def get_eval_dataset(eval_dataset: str) -> str:
    if eval_dataset.startswith("global_mmlu_"):
        return "global_mmlu"
    elif eval_dataset.startswith("truthfulqa_"):
        return "truthfulqa"
    elif eval_dataset.startswith("arc_"):
        return "arc_challenge"
    elif eval_dataset.startswith("hellaswag_"):
        return "hellaswag"
    else:
        raise ValueError(
            f"Unable to extract eval_dataset from eval_dataset: {eval_dataset}"
        )


def extract_scores(
    result_obj: Dict[str, Any]
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    score = result_obj.get("acc,none")
    score_stderr = result_obj.get("acc_stderr,none")

    score_norm = result_obj.get("acc_norm,none")
    score_norm_stderr = result_obj.get("acc_norm_stderr,none")

    if score_norm is None or score_norm_stderr is None:
        score_norm = score
        score_norm_stderr = score_stderr

    return score, score_stderr, score_norm, score_norm_stderr


def process_file(path: str, eval_mode: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)

    configs = data.get("configs", {}) or {}
    results = data.get("results", {}) or {}
    group_subtasks = data.get("group_subtasks", {}) or {}

    model_name, ft_dataset, ft_lang = extract_ft_info(configs, eval_mode=eval_mode)
    model_size = extract_model_size(model_name)
    seed = extract_seed_from_configs(configs)

    subtask_names = {sub for subs in group_subtasks.values() for sub in subs}

    rows: List[Dict[str, Any]] = []
    for eval_dataset_key, res in results.items():
        if eval_dataset_key in subtask_names:
            continue

        score, score_stderr, score_norm, score_norm_stderr = extract_scores(res)
        eval_language = get_eval_language(eval_dataset_key)
        eval_dataset = get_eval_dataset(eval_dataset_key)

        rows.append(
            {
                "model_name": model_name,
                "model_size": model_size,
                "seed": seed,
                "fine_tuned_dataset": ft_dataset,
                "fine_tuned_language": ft_lang,
                "eval_dataset": eval_dataset,
                "eval_language": eval_language,
                "score": score,
                "score_stderr": score_stderr,
                "score_norm": score_norm,
                "score_norm_stderr": score_norm_stderr,
            }
        )

    return rows


def json_files_under(d: str) -> List[str]:
    return sorted(glob(os.path.join(d, "**", "*.json"), recursive=True))


def write_seed_csv(seed_dir: str, output_dir: str) -> str:
    seed_name = os.path.basename(seed_dir.rstrip("/"))  # e.g., "seed17"
    base_dir = os.path.join(seed_dir, "base_model")
    adapter_dir = os.path.join(seed_dir, "fine_tuned")

    if not os.path.isdir(base_dir):
        raise SystemExit(f"Missing base_model dir: {base_dir}")
    if not os.path.isdir(adapter_dir):
        raise SystemExit(f"Missing fine_tuned dir: {adapter_dir}")

    base_files = json_files_under(base_dir)
    adapter_files = json_files_under(adapter_dir)

    if not base_files:
        raise SystemExit(f"No JSON files found in: {base_dir}")
    if not adapter_files:
        raise SystemExit(f"No JSON files found in: {adapter_dir}")

    fieldnames = [
        "model_name",
        "model_size",
        "seed",
        "fine_tuned_dataset",
        "fine_tuned_language",
        "eval_dataset",
        "eval_language",
        "score",
        "score_stderr",
        "score_norm",
        "score_norm_stderr",
    ]
    key_fields = [
        "model_name",
        "model_size",
        "seed",
        "fine_tuned_dataset",
        "fine_tuned_language",
        "eval_dataset",
        "eval_language",
    ]

    all_rows: List[Dict[str, Any]] = []
    for jf in base_files:
        all_rows.extend(process_file(jf, eval_mode="base"))
    for jf in adapter_files:
        all_rows.extend(process_file(jf, eval_mode="adapter"))

    unique_rows: List[Dict[str, Any]] = []
    seen = set()
    for row in all_rows:
        key = tuple(row[field] for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(
        output_dir, f"results_{seed_name}.csv"
    )  # seed17.csv, seed42.csv, ...
    with open(out_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(unique_rows)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", default="./data/lm-eval", help="Root containing seed*/ dirs."
    )
    parser.add_argument(
        "--output_dir", default="./output/data", help="Where to write seed CSVs."
    )
    args = parser.parse_args()

    seed_dirs = sorted(
        d for d in glob(os.path.join(args.root_dir, "*")) if os.path.isdir(d)
    )
    if not seed_dirs:
        raise SystemExit(f"No seed directories found under: {args.root_dir}")

    for sd in seed_dirs:
        out_csv = write_seed_csv(sd, args.output_dir)
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
