from dataclasses import dataclass
import random
import os

from datasets import load_dataset, concatenate_datasets, Dataset
import numpy as np

kwargs = {"trust_remote_code": True, "num_proc": min(4, os.cpu_count() - 4)}

##############
# PAIRS
##############


@dataclass
class MultilingualNLIEntailmentPairs:
    dataset = load_dataset("LazarusNLP/multilingual-NLI-26lang-2mil7-id", split="train", **kwargs)
    # filter for entailment pairs
    dataset = dataset.filter(lambda example: example["label"] == 0)
    dataset = dataset.select_columns(["premise", "hypothesis"])


@dataclass
class WReTEEntailmentPairs:
    dataset = load_dataset("SEACrowd/wrete", split="train", **kwargs)
    # filter for entailment pairs
    dataset = dataset.filter(lambda example: example["label"] == "Entail_or_Paraphrase")
    dataset = dataset.select_columns(["sent_A", "sent_B"])


@dataclass
class IndoLEMNTPEntailmentPairs:
    dataset = load_dataset("SEACrowd/indolem_ntp", split="train", **kwargs)
    # filter for entailment pairs
    dataset = dataset.filter(lambda example: example["label"] == 1)
    dataset = dataset.select_columns(["tweets", "next_tweet"])


@dataclass
class TyDiQA:
    dataset = load_dataset("khalidalt/tydiqa-goldp", "indonesian", split="train", **kwargs)
    dataset_1 = dataset.map(
        lambda question, passage: {"query": question, "positive": passage},
        input_columns=["question_text", "passage_text"],
    )
    dataset_2 = dataset.map(
        lambda question, answers: {"query": question, "positive": answers["text"][0]},
        input_columns=["question_text", "answers"],
    )
    dataset = concatenate_datasets([dataset_1, dataset_2]).select_columns(["query", "positive"])


@dataclass
class FacQA:
    dataset = load_dataset("SEACrowd/facqa", split="train", **kwargs)
    dataset_1 = dataset.map(
        lambda question, passage: {"query": " ".join(question), "positive": " ".join(passage)},
        input_columns=["question", "passage"],
    )
    dataset_2 = dataset.map(
        lambda question, passage, seq_label: {
            "query": " ".join(question),
            "positive": " ".join(t for t, l in zip(passage, seq_label) if l != "O"),
        },
        input_columns=["question", "passage", "seq_label"],
    )
    dataset = concatenate_datasets([dataset_1, dataset_2]).select_columns(["query", "positive"])


@dataclass
class LFQAID:
    dataset = load_dataset("indonesian-nlp/lfqa_id", split="train", **kwargs)
    dataset = dataset.map(
        lambda title, answers: {"question": title, "answers": answers["text"][np.argmax(answers["score"])]},
        input_columns=["title", "answers"],
    )
    dataset = dataset.select_columns(["question", "answers"])


@dataclass
class IndoQA:
    dataset = load_dataset("jakartaresearch/indoqa", split="train", **kwargs)
    dataset_1 = dataset.map(
        lambda question, context: {"query": question, "positive": context},
        input_columns=["question", "context"],
    )
    dataset_2 = dataset.map(
        lambda question, answer: {"query": question, "positive": answer},
        input_columns=["question", "answer"],
    )
    dataset = concatenate_datasets([dataset_1, dataset_2]).select_columns(["query", "positive"])


@dataclass
class ParaphraseDetection:
    dataset = load_dataset("jakartaresearch/id-paraphrase-detection", split="train", **kwargs)
    dataset = dataset.select_columns(["sentence1", "sentence2"])


@dataclass
class Wikipedia:
    dataset = load_dataset("wikimedia/wikipedia", "20231101.id", split="train", **kwargs)
    dataset = dataset.select_columns(["text", "title"])


@dataclass
class Brainly:
    dataset = load_dataset("lesserfield/brainly", split="train", **kwargs)
    dataset = dataset.filter(lambda status: status == "verified", input_columns=["status_1"])
    dataset = dataset.select_columns(["instruction", "answer_1"])


@dataclass
class IndonesianNews:
    dataset = load_dataset("esteler-ai/idn-news-az", split="train", **kwargs)
    dataset = dataset.select_columns(["text", "title"])


@dataclass
class DoctorQA:
    dataset = load_dataset("hermanshid/doctor-id-qa", split="train", **kwargs)
    dataset = dataset.select_columns(["question", "answer"])


@dataclass
class Liputan6:
    dataset = load_dataset("SEACrowd/liputan6", split="train", **kwargs)
    dataset = dataset.select_columns(["document", "summary"])


##############
# TRIPLETS
##############


@dataclass
class mMARCO:
    dataset = load_dataset("unicamp-dl/mmarco", "indonesian", split="train", **kwargs)
    dataset = dataset.shuffle(seed=42).select(range(100_000))  # limit to only 100,000 rows
    dataset = dataset.select_columns(["query", "positive", "negative"])


@dataclass
class MIRACL:
    dataset = load_dataset("miracl/miracl", "id", split="train", **kwargs)
    dataset = dataset.filter(lambda negatives: len(negatives) > 0, input_columns=["negative_passages"])
    dataset = dataset.map(
        lambda query, positives, negatives: {
            "query": query,
            "positives": [p["text"] for p in random.sample(positives, min(2, len(positives)))],
            "negatives": [n["text"] for n in random.sample(negatives, min(2, len(negatives)))],
        },
        input_columns=["query", "positive_passages", "negative_passages"],
    )
    dataset_1 = dataset.map(
        lambda query, positives, negatives: {"query": query, "positive": positives[0], "negative": negatives[0]},
        input_columns=["query", "positives", "negatives"],
    )
    dataset_2 = dataset.map(
        lambda query, positives, negatives: {"query": query, "positive": positives[-1], "negative": negatives[-1]},
        input_columns=["query", "positives", "negatives"],
    )
    dataset = concatenate_datasets([dataset_1, dataset_2]).select_columns(["query", "positive", "negative"])


@dataclass
class SwimIR:
    dataset = load_dataset("nthakur/swim-ir-monolingual", "id", split="train", **kwargs)

    def __post_init__(self):
        columns = ["anchor", "positive", "negative"]
        train_samples = self.train_samples()
        self.dataset = Dataset.from_list([dict(zip(columns, sample)) for sample in train_samples])

    def train_samples(self) -> Dataset:
        train_data = {}
        train_samples = []

        for datum in self.dataset:
            query = datum["query"].strip()
            answer = datum["text"].strip()
            title = datum["title"].strip()

            if title not in train_data:
                train_data[title] = {query: [answer]}
            elif title in train_data and query not in train_data[title]:
                train_data[title][query] = [answer]
            else:
                train_data[title][query].append(answer)

        for title, queries in train_data.items():
            passage_queries = list(queries.keys())
            # cannot get a negative sample if only 1 query in that passage
            if len(passage_queries) > 1:
                for query, answers in queries.items():
                    positive = random.choice(answers)
                    # get random negative sample, from different query
                    random_query = random.choice([q for q in passage_queries if q != query])
                    negative = random.choice(queries[random_query])
                    train_samples.append([query, positive, negative])

        return train_samples


@dataclass
class IndoStoryCloze:
    dataset = load_dataset("indolem/indo_story_cloze", split="train", **kwargs)
    dataset = dataset.map(
        lambda s1, s2, s3, s4: {"context": ". ".join([s1, s2, s3, s4])},
        input_columns=["sentence-1", "sentence-2", "sentence-3", "sentence-4"],
    )
    dataset = dataset.select_columns(["context", "correct_ending", "incorrect_ending"])


@dataclass
class IndoNLI:
    dataset = load_dataset("indonli", split="train", **kwargs)

    def __post_init__(self):
        columns = ["premise", "hypothesis", "contradiction"]
        train_samples = self.train_samples()
        self.dataset = Dataset.from_list([dict(zip(columns, sample)) for sample in train_samples])

    def train_samples(self) -> Dataset:
        id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
        train_data = {}
        train_samples = []

        def add_to_samples(sent1, sent2, label):
            if sent1 not in train_data:
                train_data[sent1] = {"contradiction": set(), "entailment": set(), "neutral": set()}
            train_data[sent1][label].add(sent2)

        for datum in self.dataset:
            sent1 = datum["premise"].strip()
            sent2 = datum["hypothesis"].strip()
            add_to_samples(sent1, sent2, id2label[datum["label"]])
            add_to_samples(sent2, sent1, id2label[datum["label"]])  # Also add the opposite

        for sent1, others in train_data.items():
            if len(others["entailment"]) > 0 and len(others["contradiction"]) > 0:
                entailments, contradictions = list(others["entailment"]), list(others["contradiction"])
                train_samples.append([sent1, random.choice(entailments), random.choice(contradictions)])
                train_samples.append([sent1, random.choice(entailments), random.choice(contradictions)])

        return train_samples
