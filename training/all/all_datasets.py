from typing import List
from dataclasses import dataclass
import random

from datasets import load_dataset
from sentence_transformers import InputExample
import numpy as np

##############
# PAIRS
##############


@dataclass
class WReTE:
    dataset = load_dataset("SEACrowd/wrete", split="train", trust_remote_code=True)
    # filter for entailment pairs
    dataset = dataset.filter(lambda example: example["label"] == "Entail_or_Paraphrase")

    @staticmethod
    def train_samples() -> List[InputExample]:
        train_samples = []

        for datum in WReTE.dataset:
            train_samples.append(InputExample(texts=[datum["sent_A"], datum["sent_B"]]))

        return train_samples


@dataclass
class IndoLEMNTP:
    dataset = load_dataset("SEACrowd/indolem_ntp", split="train", trust_remote_code=True)
    # filter for entailment pairs
    dataset = dataset.filter(lambda example: example["label"] == 1)

    @staticmethod
    def train_samples() -> List[InputExample]:
        train_samples = []

        for datum in IndoLEMNTP.dataset:
            train_samples.append(InputExample(texts=[datum["tweets"], datum["next_tweet"]]))

        return train_samples


@dataclass
class TyDiQA:
    dataset = load_dataset("khalidalt/tydiqa-goldp", "indonesian", split="train", trust_remote_code=True).shuffle(
        seed=42
    )

    @staticmethod
    def train_samples() -> List[InputExample]:
        train_samples = []

        for datum in TyDiQA.dataset:
            train_samples.append(InputExample(texts=[datum["question_text"], datum["passage_text"]]))
            train_samples.append(InputExample(texts=[datum["question_text"], datum["answers"]["text"][0]]))

        return train_samples


@dataclass
class FacQA:
    dataset = load_dataset("SEACrowd/facqa", split="train", trust_remote_code=True)

    @staticmethod
    def train_samples() -> List[InputExample]:
        train_samples = []

        for datum in FacQA.dataset:
            question = " ".join(datum["question"])
            passage = " ".join(datum["passage"])
            answer = " ".join(t for t, l in zip(datum["passage"], datum["seq_label"]) if l != "O")

            train_samples.append(InputExample(texts=[question, passage]))
            train_samples.append(InputExample(texts=[question, answer]))

        return train_samples


@dataclass
class LFQAID:
    dataset = load_dataset("indonesian-nlp/lfqa_id", split="train", trust_remote_code=True)

    @staticmethod
    def train_samples() -> List[InputExample]:
        train_samples = []

        for datum in LFQAID.dataset:
            question = datum["title"]
            scores = datum["answers"]["score"]
            answer = datum["answers"]["text"][np.argmax(scores)]

            train_samples.append(InputExample(texts=[question, answer]))

        return train_samples


@dataclass
class IndoQA:
    dataset = load_dataset("jakartaresearch/indoqa", split="train", trust_remote_code=True)

    @staticmethod
    def train_samples() -> List[InputExample]:
        train_samples = []

        for datum in IndoQA.dataset:
            question = datum["question"]
            passage = datum["context"]
            answer = datum["answer"]

            if question and passage and answer:
                train_samples.append(InputExample(texts=[question, passage]))
                train_samples.append(InputExample(texts=[question, answer]))

        return train_samples


@dataclass
class ParaphraseDetection:
    dataset = load_dataset("jakartaresearch/id-paraphrase-detection", split="train", trust_remote_code=True)

    @staticmethod
    def train_samples() -> List[InputExample]:
        train_samples = []

        for datum in ParaphraseDetection.dataset:
            train_samples.append(InputExample(texts=[datum["sentence1"], datum["sentence2"]]))

        return train_samples


##############
# TRIPLETS
##############


@dataclass
class mMARCO:
    dataset = load_dataset("unicamp-dl/mmarco", "indonesian", split="train", trust_remote_code=True)
    # limit to only 100,000 rows
    dataset = dataset.shuffle(seed=42).select(range(100_000))

    @staticmethod
    def train_samples() -> List[InputExample]:
        train_samples = []

        for datum in mMARCO.dataset:
            train_samples.append(
                InputExample(
                    texts=[
                        datum["query"],
                        datum["positive"],
                        datum["negative"],
                    ]
                )
            )

        return train_samples


@dataclass
class MIRACL:
    dataset = load_dataset("miracl/miracl", "id", split="train", trust_remote_code=True)

    @staticmethod
    def train_samples() -> List[InputExample]:
        train_samples = []

        for datum in MIRACL.dataset:
            query = datum["query"]
            positives = [doc["text"] for doc in datum["positive_passages"]]
            negatives = [doc["text"] for doc in datum["negative_passages"]]

            if len(negatives) > 0:
                train_samples.append(InputExample(texts=[query, random.choice(positives), random.choice(negatives)]))
                train_samples.append(InputExample(texts=[random.choice(positives), query, random.choice(negatives)]))

        return train_samples


@dataclass
class IndoStoryCloze:
    dataset = load_dataset("indolem/indo_story_cloze", split="train", trust_remote_code=True)

    @staticmethod
    def train_samples() -> List[InputExample]:
        train_samples = []

        for datum in IndoStoryCloze.dataset:
            context = ". ".join([datum["sentence-1"], datum["sentence-2"], datum["sentence-3"], datum["sentence-4"]])
            train_samples.append(
                InputExample(
                    texts=[
                        context,
                        datum["correct_ending"],
                        datum["incorrect_ending"],
                    ]
                )
            )

        return train_samples


@dataclass
class IndoNLI:
    dataset = load_dataset("indonli", split="train", trust_remote_code=True)
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}

    @staticmethod
    def train_samples() -> List[InputExample]:
        def add_to_samples(sent1, sent2, label):
            if sent1 not in train_data:
                train_data[sent1] = {"contradiction": set(), "entailment": set(), "neutral": set()}
            train_data[sent1][label].add(sent2)

        train_data = {}
        train_samples = []

        for datum in IndoNLI.dataset:
            sent1 = datum["premise"].strip()
            sent2 = datum["hypothesis"].strip()

            add_to_samples(sent1, sent2, IndoNLI.id2label[datum["label"]])
            add_to_samples(sent2, sent1, IndoNLI.id2label[datum["label"]])  # Also add the opposite

        for sent1, others in train_data.items():
            if len(others["entailment"]) > 0 and len(others["contradiction"]) > 0:
                train_samples.append(
                    InputExample(
                        texts=[
                            sent1,
                            random.choice(list(others["entailment"])),
                            random.choice(list(others["contradiction"])),
                        ]
                    )
                )
                train_samples.append(
                    InputExample(
                        texts=[
                            random.choice(list(others["entailment"])),
                            sent1,
                            random.choice(list(others["contradiction"])),
                        ]
                    )
                )

        return train_samples
