from abc import ABC, abstractmethod
import os
from pathlib import Path

from datasets import load_dataset, load_from_disk


def load_aligned_dataset(benchmark, hub_dataset, language):
    """Load a local manifest-built dataset when DONORS_DATA_ROOT is set."""
    local_root = os.getenv("DONORS_DATA_ROOT")
    if local_root:
        path = Path(local_root) / benchmark / language
        if not path.is_dir():
            raise FileNotFoundError(
                f"Local aligned dataset not found: {path}. "
                "Run `make materialize-datasets` first."
            )
        return load_from_disk(path)
    return load_dataset(hub_dataset, language)


class BaseDatasetProcessor(ABC):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @abstractmethod
    def load_n_preprocess_dataset(self, language):
        """Preprocess the dataset for fine-tuning"""
        pass

    @staticmethod
    @abstractmethod
    def build_tasks(languages):
        """Build tasks list for evaluation"""
        pass

    @abstractmethod
    def get_train_dataset(self, dataset):
        """
        Returns a train_dataset
        """
        pass

    def process(self, language):
        """Load and preprocess dataset for a specific language"""
        processed_dataset = self.load_n_preprocess_dataset(language)
        return self.get_train_dataset(processed_dataset)


def get_processor(dataset_name):
    """Factory function to get the appropriate dataset processor based on the dataset name in dynamic way.
    This solves circular imports."""
    if dataset_name == "arc_challenge":
        from .arc_challenge import ARCChallengeProcessor

        return ARCChallengeProcessor
    elif dataset_name == "global_mmlu":
        from .global_mmlu import GlobalMMLUProcessor

        return GlobalMMLUProcessor
    elif dataset_name == "truthfulqa":
        from .truthfulqa import TruthfulQAProcessor

        return TruthfulQAProcessor

    elif dataset_name == "hellaswag":
        from .hellaswag import HellaSwagProcessor

        return HellaSwagProcessor
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
