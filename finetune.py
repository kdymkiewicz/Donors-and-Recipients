import argparse
import os
from typing import Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    TrainingArguments,
    Trainer,
    set_seed,
)

from dataset_processors.base import get_processor
from utils import load_tokenizer, load_base_model, get_clean_model_name, get_report_to


TRAIN_VALIDATION_SPLIT_SEED = 42
EXPERIMENT_SEEDS = (17, 42, 101)


def add_lora_adapters(model, r: int = 32, alpha: int | None = None):
    """
    Attach LoRA adapters to the model.
    """
    if alpha is None:
        alpha = 2 * r

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    return model


def load_dataset(dataset_name: str, language: str, tokenizer):
    """
    Get the training dataset via dataset_processors.
    """
    processor_cls = get_processor(dataset_name)
    processor = processor_cls(tokenizer)
    ds = processor.process(language)
    return ds


def save_lora_adapter(lora_model, adapter_dir: str):
    """
    Save the LoRA adapter weights/config (PEFT format).
    """
    os.makedirs(adapter_dir, exist_ok=True)
    print(f"[save] Saving LoRA adapter to {adapter_dir} ...")
    lora_model.save_pretrained(adapter_dir)
    print("[save] LoRA adapter saved.")


def merge_and_save_full_model(lora_model, tokenizer, save_dir: str):
    """
    Merge LoRA into base model and save as a standalone model.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"[merge] Merging LoRA adapters and saving full model to {save_dir} ...")
    merged_model = lora_model.merge_and_unload()
    merged_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("[merge] Full model saved.")


def fine_tune_model(
    base_model_name: str,
    dataset_name: str,
    language: str,
    output_dir: str,
    run_name: str,
    experiment_seed: int,
) -> Tuple[torch.nn.Module, "PreTrainedTokenizer"]:
    """
    LoRA fine-tuning run for a single (model, dataset, language).
    """
    set_seed(experiment_seed)

    # Load base model + tokenizer
    model = load_base_model(base_model_name)
    tokenizer = load_tokenizer(base_model_name)

    # Attach LoRA
    model = add_lora_adapters(model)

    # Load train dataset
    ds = load_dataset(dataset_name, language, tokenizer)
    split = ds.train_test_split(test_size=0.1, seed=TRAIN_VALIDATION_SPLIT_SEED)
    train_ds, val_ds = split["train"], split["test"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=3.0,
        auto_find_batch_size=True,
        bf16=True,
        bf16_full_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        save_total_limit=3,
        warmup_ratio=0.1,
        prediction_loss_only=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=get_report_to(),
        seed=experiment_seed,
        data_seed=experiment_seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    trainer.train()

    return trainer.model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for a single dataset:language source."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Language code",
    )
    parser.add_argument(
        "--experiment_seed",
        type=int,
        choices=EXPERIMENT_SEEDS,
        required=True,
        help="Training seed (17, 42, or 101); the aligned dataset split is fixed.",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        required=True,
        help="Directory root for Trainer checkpoints/logs.",
    )
    parser.add_argument(
        "--lora_adapters_dir",
        type=str,
        required=True,
        help="Directory root for saved LoRA adapters.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    clean_model_name = get_clean_model_name(args.model_name)
    dataset_name = args.dataset
    language = args.language
    experiment_seed = args.experiment_seed

    run_name = (
        f"ft__{clean_model_name}__{dataset_name}__{language}"
        f"__seed{experiment_seed}"
    )

    ckpt_dir = os.path.join(
        args.checkpoints_dir,
        dataset_name,
        language,
        clean_model_name,
        f"seed{experiment_seed}",
    )
    adapter_dir = os.path.join(
        args.lora_adapters_dir,
        dataset_name,
        language,
        clean_model_name,
        f"seed{experiment_seed}",
    )

    print(
        f"[run] model={args.model_name} "
        f"dataset={dataset_name} language={language} "
        f"experiment_seed={experiment_seed}\n"
        f"[run] validation_split_seed={TRAIN_VALIDATION_SPLIT_SEED}\n"
        f"[run] checkpoints_dir={ckpt_dir}\n"
        f"[run] adapter_dir={adapter_dir}\n"
    )

    # Fine-tune with fixed step budget
    lora_model, tokenizer = fine_tune_model(
        base_model_name=args.model_name,
        dataset_name=dataset_name,
        language=language,
        output_dir=ckpt_dir,
        run_name=run_name,
        experiment_seed=experiment_seed,
    )

    # Save adapter-only weights
    save_lora_adapter(lora_model, adapter_dir)


if __name__ == "__main__":
    main()
