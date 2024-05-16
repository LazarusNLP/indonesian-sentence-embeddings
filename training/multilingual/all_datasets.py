from typing import List
from dataclasses import dataclass

from datasets import load_dataset, concatenate_datasets


@dataclass
class NusaX:
    target_languages = ["ace", "ban", "bjn", "bbc", "bug", "jav", "mad", "min", "nij", "sun"]
    train_datasets, validation_datasets = [], []
    for lang in target_languages:
        ds = load_dataset("indonlp/NusaX-MT", f"ind-{lang}", trust_remote_code=True)
        train_datasets.append(ds["train"])
        validation_datasets.append(ds["validation"])

    dataset = {"train": concatenate_datasets(train_datasets), "validation": concatenate_datasets(validation_datasets)}

    @staticmethod
    def train_samples() -> List[List[str]]:
        train_samples = []

        for datum in NusaX.dataset["train"]:
            train_samples.append([datum["text_1"], datum["text_2"]])

        return train_samples

    @staticmethod
    def validation_samples() -> List[List[str]]:
        validation_samples = []

        for datum in NusaX.dataset["validation"]:
            validation_samples.append([datum["text_1"], datum["text_2"]])

        return validation_samples


@dataclass
class NusaTranslation:
    target_languages = ["abs", "btk", "bew", "bhp", "jav", "mad", "mak", "min", "mui", "rej", "sun"]
    train_datasets, validation_datasets = [], []
    for lang in target_languages:
        ds = load_dataset(
            "indonlp/nusatranslation_mt", f"nusatranslation_mt_ind_{lang}_nusantara_t2t", trust_remote_code=True
        )
        train_datasets.append(ds["train"])
        validation_datasets.append(ds["validation"])

    dataset = {"train": concatenate_datasets(train_datasets), "validation": concatenate_datasets(validation_datasets)}

    @staticmethod
    def train_samples() -> List[List[str]]:
        train_samples = []

        for datum in NusaTranslation.dataset["train"]:
            train_samples.append([datum["text_1"], datum["text_2"]])

        return train_samples

    @staticmethod
    def validation_samples() -> List[List[str]]:
        validation_samples = []

        for datum in NusaTranslation.dataset["validation"]:
            validation_samples.append([datum["text_1"], datum["text_2"]])

        return validation_samples
