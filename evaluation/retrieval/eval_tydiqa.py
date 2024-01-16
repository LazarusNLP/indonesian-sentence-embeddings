from dataclasses import dataclass
import os

from datargs import parse
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator


@dataclass
class Args:
    model_name: str = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    test_dataset_name: str = "khalidalt/tydiqa-goldp"
    test_dataset_config: str = "indonesian"
    test_dataset_split: str = "validation"
    test_batch_size: int = 32
    output_folder: str = "results"


def main(args: Args):
    os.makedirs(args.output_folder, exist_ok=True)

    model = SentenceTransformer(args.model_name)

    # Load dataset
    test_ds = load_dataset(args.test_dataset_name, args.test_dataset_config, split=args.test_dataset_split)

    # Get all queries and documents
    queries = test_ds["question_text"]
    documents = list(set(test_ds["passage_text"]))
    doc2idx = {d: i for i, d in enumerate(documents)}

    # Map index of query to set of relevant context documents
    relevant_docs = {idx: set([doc2idx[data["passage_text"]]]) for idx, data in enumerate(test_ds)}

    # Assign IDs to queries and context documents
    queries = dict(enumerate(queries))
    corpus = dict(enumerate(documents))

    # Evaluate
    evaluator = InformationRetrievalEvaluator(queries, corpus, relevant_docs, batch_size=args.test_batch_size)
    evaluator(model, output_path=args.output_folder)


if __name__ == "__main__":
    args = parse(Args)
    main(args)
