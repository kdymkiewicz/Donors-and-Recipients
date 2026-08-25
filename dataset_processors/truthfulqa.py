from .base import BaseDatasetProcessor, load_aligned_dataset


class TruthfulQAProcessor(BaseDatasetProcessor):

    def get_train_dataset(self, dataset):
        return dataset["train"]

    @staticmethod
    def build_tasks(languages):
        tasks = []
        for lang in languages:
            tasks.extend([f"truthfulqa_{lang}_mc1"])
        return tasks

    @staticmethod
    def _load_dataset(language):
        return load_aligned_dataset(
            "truthfulqa",
            "Dr4kl3s/truthfulqa_coregrid_seed42",
            language,
        )

    def load_n_preprocess_dataset(self, language):
        dataset = self._load_dataset(language)

        processed_dataset = {}
        for split in dataset:
            original_columns = dataset[split].column_names
            processed_dataset[split] = dataset[split].map(
                self._tokenize,
                batched=True,
                remove_columns=original_columns,
                desc=f"Tokenizing truthfulqa (mc1) {language} [{split}]",
            )

        return processed_dataset

    def _tokenize(self, examples):
        max_length = 128

        texts = []

        for question, choices, labels in zip(
            examples["question"],
            examples["mc1_targets_choices"],
            examples["mc1_targets_labels"],
        ):

            correct_indices = [i for i, lab in enumerate(labels) if lab == 1]
            idx = correct_indices[0]
            answer = choices[idx]
            texts.append(f"Question: {question}\nAnswer: {answer}")

        if len(texts) == 0:
            return {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }

        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        labels = []
        for ids, mask in zip(input_ids, attention_mask):
            labs = [tok if m == 1 else -100 for tok, m in zip(ids, mask)]
            labels.append(labs)

        enc["labels"] = labels
        return enc
