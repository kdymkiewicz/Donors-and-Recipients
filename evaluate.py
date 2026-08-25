import argparse
import os
import shutil
import subprocess
import sys

from utils import get_clean_model_name, merge_lora_and_save, gather_tasks


EXPERIMENT_SEEDS = (17, 42, 101)
HARNESS_SEEDS = "0,1234,1234,1234"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a base model or a merged LoRA adapter."
    )

    parser.add_argument(
        "--eval_mode",
        choices=["base", "adapter"],
        required=True,
        help="Evaluate the base HF model directly or merge a LoRA adapter first.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model id or local path to the base model.",
    )
    parser.add_argument(
        "--dataset",
        help="Dataset name (required for adapter mode).",
    )
    parser.add_argument(
        "--language",
        help="Language code (required for adapter mode).",
    )
    parser.add_argument(
        "--experiment_seed",
        type=int,
        choices=EXPERIMENT_SEEDS,
        required=True,
        help="Adapter training seed; the aligned dataset split is fixed.",
    )
    parser.add_argument(
        "--lora_adapters_dir",
        help="Root directory containing saved LoRA adapters.",
    )
    parser.add_argument(
        "--merged_models_dir",
        help="Root directory to place temporary merged models.",
    )
    parser.add_argument(
        "--outputs_dir",
        required=True,
        help="Directory for lm_eval JSON outputs.",
    )

    return parser.parse_args()


def build_tasks_string() -> str:
    tasks = gather_tasks()
    return ",".join(tasks)


def run_lm_eval(
    model_args: str, tasks: str, outputs_dir: str, output_name: str
) -> None:
    os.makedirs(outputs_dir, exist_ok=True)
    output_path = os.path.join(outputs_dir, output_name)

    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        tasks,
        "--batch_size",
        "auto",
        "--apply_chat_template",
        "--num_fewshot",
        "0",
        "--confirm_run_unsafe_code",
        "--seed",
        HARNESS_SEEDS,
        "--output_path",
        output_path,
    ]

    print("[eval] Running lm_eval:\n  " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def eval_base_model(args: argparse.Namespace) -> None:
    clean_model_name = get_clean_model_name(args.model)
    tasks = build_tasks_string()

    model_args = f"pretrained={args.model}"
    output_name = (
        f"lm_eval_base_model_{clean_model_name}_seed{args.experiment_seed}.json"
    )

    run_lm_eval(
        model_args=model_args,
        tasks=tasks,
        outputs_dir=args.outputs_dir,
        output_name=output_name,
    )


def eval_adapter(args: argparse.Namespace) -> None:
    if not args.dataset or not args.language:
        raise ValueError("--dataset and --language are required for adapter eval_mode")
    if not args.lora_adapters_dir or not args.merged_models_dir:
        raise ValueError(
            "--lora_adapters_dir and --merged_models_dir are required for adapter mode"
        )

    clean_model_name = get_clean_model_name(args.model)
    adapter_path = os.path.join(
        args.lora_adapters_dir,
        args.dataset,
        args.language,
        clean_model_name,
        f"seed{args.experiment_seed}",
    )

    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    merged_model_dir = os.path.join(
        args.merged_models_dir,
        args.dataset,
        args.language,
        clean_model_name,
        f"seed{args.experiment_seed}",
    )
    os.makedirs(os.path.dirname(merged_model_dir), exist_ok=True)

    print(
        f"[merge_eval] Merging adapter:\n  base:   {args.model}\n  adapter:{adapter_path}\n  out:    {merged_model_dir}",
        flush=True,
    )

    merge_lora_and_save(
        base_model_name=args.model,
        lora_adapter_path=adapter_path,
        output_path=merged_model_dir,
    )

    tasks = build_tasks_string()

    model_args = f"pretrained={merged_model_dir},tokenizer={os.path.join(merged_model_dir, 'tokenizer')}"
    output_name = (
        f"lm_eval_ft_model_{clean_model_name}_{args.dataset}_{args.language}"
        f"_seed{args.experiment_seed}.json"
    )

    run_lm_eval(
        model_args=model_args,
        tasks=tasks,
        outputs_dir=args.outputs_dir,
        output_name=output_name,
    )

    shutil.rmtree(merged_model_dir)


def main() -> None:
    args = parse_args()

    if args.eval_mode == "base":
        eval_base_model(args)
    else:
        eval_adapter(args)


if __name__ == "__main__":
    main()
