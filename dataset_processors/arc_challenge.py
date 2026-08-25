from .base import BaseDatasetProcessor, load_aligned_dataset


class ARCChallengeProcessor(BaseDatasetProcessor):

    def get_train_dataset(self, dataset):
        return dataset["train"]

    @staticmethod
    def build_tasks(languages):
        tasks = []
        for lang in languages:
            tasks.append(f"arc_{lang}")
        return tasks

    @staticmethod
    def _load_dataset(language):
        return load_aligned_dataset(
            "arc_challenge",
            "Dr4kl3s/arc_challenge_core_grid_seed42",
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
                desc=f"Tokenizing arc_challenge {language} [{split}]",
            )

        return processed_dataset

    def tokenize(self, examples):
        max_length = 256

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []

        for q, oa, ob, oc, od, ans_raw in zip(
            examples["instruction"],
            examples["option_a"],
            examples["option_b"],
            examples["option_c"],
            examples["option_d"],
            examples["answer"],
        ):
            if ans_raw is None:
                continue
            ans = str(ans_raw).strip().upper()
            if ans not in ("A", "B", "C", "D"):
                continue

            prompt = (
                f"Question: {q.strip()}\n"
                f"A) {oa}\n"
                f"B) {ob}\n"
                f"C) {oc}\n"
                f"D) {od}\n"
                "Answer:"
            )
            target = " " + ans

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
