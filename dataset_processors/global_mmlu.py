from .base import BaseDatasetProcessor, load_aligned_dataset


class GlobalMMLUProcessor(BaseDatasetProcessor):

    def get_train_dataset(self, dataset):
        return dataset["train"]

    @staticmethod
    def build_tasks(languages):
        return [f"global_mmlu_{lang}" for lang in languages]

    @staticmethod
    def _load_dataset(language):
        return load_aligned_dataset(
            "global_mmlu",
            "Dr4kl3s/global_mmlu_lite_core_grid_seed42",
            language,
        )

    def load_n_preprocess_dataset(self, language):
        dataset = self._load_dataset(language)

        processed_dataset = {}
        for split in dataset:
            original_columns = dataset[split].column_names
            processed_dataset[split] = dataset[split].map(
                self._tokenize_function, batched=True, remove_columns=original_columns
            )

        return processed_dataset

    def _tokenize_function(self, examples):
        max_length = 512

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        idx2letter = {0: "A", 1: "B", 2: "C", 3: "D"}

        for q, a, b, c, d, ans_raw in zip(
            examples["question"],
            examples["option_a"],
            examples["option_b"],
            examples["option_c"],
            examples["option_d"],
            examples["answer"],
        ):
            if isinstance(ans_raw, int):
                ans = idx2letter.get(ans_raw)
            else:
                ans = str(ans_raw).strip().upper()

            prompt = (
                f"{q.strip()}\n"
                f"A. {a}\n"
                f"B. {b}\n"
                f"C. {c}\n"
                f"D. {d}\n"
                f"Answer:"
            )
            target = " " + ans

            prompt_enc = self.tokenizer(
                prompt,
                add_special_tokens=False,
            )
            target_enc = self.tokenizer(
                target,
                add_special_tokens=False,
            )

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
