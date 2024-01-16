from dataclasses import dataclass
import os

from datargs import parse
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator


@dataclass
class Args:
    model_name: str = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    test_dataset_name: str = "miracl/miracl"
    test_dataset_config: str = "id"
    test_dataset_split: str = "dev"
    test_batch_size: int = 32
    output_folder: str = "results"


def main(args: Args):
    os.makedirs(args.output_folder, exist_ok=True)

    model = SentenceTransformer(args.model_name)

    # Load dataset
    test_ds = load_dataset(args.test_dataset_name, args.test_dataset_config, split=args.test_dataset_split)

    # Preprocess datasets
    queries, answers, documents = [], [], []
    for data in test_ds:
        query = data["query"]
        positive_passages = [d["text"] for d in data["positive_passages"]]
        negative_passages = [d["text"] for d in data["negative_passages"]]

        # queries and positive passages are pairs
        queries.append(query)
        answers.append(positive_passages)

        documents += positive_passages
        documents += negative_passages

    # Unique-ify documents
    documents = list(set(documents))

    # Map index of query/answer to set of relevant context documents
    relevant_docs = {idx: set(documents.index(a) for a in answer) for idx, answer in enumerate(answers)}

    # Assign IDs to queries and context documents
    queries = dict(enumerate(queries))
    corpus = dict(enumerate(documents))

    # Evaluate
    evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs, batch_size=args.test_batch_size)
    evaluator(model, output_path=args.output_folder)


if __name__ == "__main__":
    args = parse(Args)
    main(args)
