import os
import shutil
from typing import Dict
from typing import List, Sequence

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from dataset_processors.base import get_processor


def load_config(config_path: str) -> Dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_clean_model_name(model_id: str) -> str:
    return model_id.split("/")[-1]


def load_base_model(model_name: str):
    """
    Load a base causal LM.
    """
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    )


def load_tokenizer(model_name: str):
    """
    Load tokenizer and ensure a pad token exists.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, fix_mistral_regex=True)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token

    return tokenizer


def merge_lora_and_save(
    base_model_name: str,
    lora_adapter_path: str,
    output_path: str,
) -> None:
    """
    Load a base model, apply a LoRA adapter, merge them, and save the result.

    Args:
        base_model_name: HuggingFace model name or path to base model
        lora_adapter_path: Path to the LoRA adapter
        output_path: Path to save the merged model
    """
    # Load base model
    base_model = load_base_model(base_model_name)

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    # Merge LoRA weights into base model
    merged_model = model.merge_and_unload()

    # Save merged model
    merged_model.save_pretrained(output_path)

    # Save tokenizer
    tokenizer = load_tokenizer(base_model_name)
    tokenizer.save_pretrained(output_path + "/tokenizer")

    return output_path


def delete_model_directory(model_path: str, confirm: bool = True) -> None:
    """
    Delete all content (model and tokenizer) from the specified directory.

    Args:
        model_path: Path to the model directory to delete
        confirm: If True, print confirmation message; if False, silently delete
    """
    if not os.path.exists(model_path):
        print(f"Directory does not exist: {model_path}")
        return

    shutil.rmtree(model_path)

    if confirm:
        print(f"Deleted model directory: {model_path}")


def get_report_to():
    # Auto-enable when launched via `wandb agent`
    if os.getenv("WANDB_SWEEP_ID") is not None:
        return ["wandb"]


CORE_GRID_DATASETS = ["arc_challenge", "global_mmlu", "truthfulqa", "hellaswag"]
CORE_GRID_LANGUAGES = [
    "ar",
    "bn",
    "de",
    "en",
    "es",
    "fr",
    "hi",
    "id",
    "it",
    "pt",
    "zh",
]


def gather_tasks(
    datasets: Sequence[str] = CORE_GRID_DATASETS,
    languages: Sequence[str] = CORE_GRID_LANGUAGES,
) -> List[str]:
    tasks: List[str] = []

    for dataset in datasets:
        try:
            processor_cls = get_processor(dataset)
        except Exception as exc:
            raise ValueError(f"Unknown dataset '{dataset}'") from exc

        dataset_tasks = processor_cls.build_tasks(languages)
        for task in dataset_tasks:
            tasks.append(task)

    return tasks
