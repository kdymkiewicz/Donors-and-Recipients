import re

from .base import BaseDatasetProcessor, load_aligned_dataset


class HellaSwagProcessor(BaseDatasetProcessor):

    def get_train_dataset(self, dataset):
        return dataset["train"]

    @staticmethod
    def build_tasks(languages):
        return [f"hellaswag_{lang}" for lang in languages]

    @staticmethod
    def _load_dataset(language):
        return load_aligned_dataset(
            "hellaswag",
            "Dr4kl3s/hellaswag_coregrid_seed42",
            language,
        )

    def load_n_preprocess_dataset(self, language):
        dataset = self._load_dataset(language)

        processed_dataset = {}
        for split in dataset:
            original_columns = dataset[split].column_names
            processed_dataset[split] = dataset[split].map(
                self.tokenize,
                batched=True,
                remove_columns=original_columns,
                load_from_cache_file=False,
            )

        return processed_dataset

    def tokenize(self, examples):
        max_length = 512

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        for ctx_a, ctx_b, activity_label, endings, label in zip(
            examples["ctx_a"],
            examples["ctx_b"],
            examples["activity_label"],
            examples["endings"],
            examples["label"],
        ):
            # ---------- build query ----------
            ctx = ctx_a + " " + ctx_b.capitalize()
            query = self.preprocess(activity_label + ": " + ctx)

            endings_clean = [self.preprocess(e) for e in endings]
            idx = int(label)

            answer_letter = ["A", "B", "C", "D"][idx]

            # ---------- build prompt + target ----------
            prompt = (
                f"{query}\n"
                f"A) {endings_clean[0]}\n"
                f"B) {endings_clean[1]}\n"
                f"C) {endings_clean[2]}\n"
                f"D) {endings_clean[3]}\n"
                "Answer:"
            )
            target = " " + answer_letter  # what we want the model to generate

            # Tokenize prompt & target separately, no special tokens
            prompt_enc = self.tokenizer(prompt, add_special_tokens=False)
            target_enc = self.tokenizer(target, add_special_tokens=False)

            prompt_ids = prompt_enc["input_ids"]
            target_ids = target_enc["input_ids"]

            ids = prompt_ids + target_ids
            attn = [1] * len(ids)

            if len(ids) > max_length:
                ids = ids[-max_length:]
                attn = attn[-max_length:]

            labels = [-100] * len(ids)
            ans_len = len(target_ids)
            start = max(0, len(ids) - ans_len)
            for j in range(start, len(ids)):
                labels[j] = ids[j]

            pad_len = max_length - len(ids)
            if pad_len > 0:
                ids = ids + [pad_token_id] * pad_len
                attn = attn + [0] * pad_len
                labels = labels + [-100] * pad_len

            input_ids_batch.append(ids)
            attention_mask_batch.append(attn)
            labels_batch.append(labels)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels_batch,
        }

    def preprocess(self, text):
        """Preprocess text to clean up WikiHow artifacts."""
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub(r"\[.*?\]", "", text)
        text = text.replace("  ", " ")
        return text
