from dataclasses import dataclass
import os

from datargs import parse
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


@dataclass
class Args:
    model_name: str = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    test_dataset_name: str = "LazarusNLP/stsb_mt_id"
    test_dataset_split: str = "test"
    test_text_column_1: str = "text_1"
    test_text_column_2: str = "text_2"
    test_label_column: str = "correlation"
    test_batch_size: int = 32
    output_folder: str = "results"


def main(args: Args):
    os.makedirs(args.output_folder, exist_ok=True)

    model = SentenceTransformer(args.model_name)

    # Load dataset
    test_ds = load_dataset(args.test_dataset_name, split=args.test_dataset_split)

    test_data = [
        InputExample(
            texts=[data[args.test_text_column_1], data[args.test_text_column_2]],
            label=float(data[args.test_label_column]) / 5.0,
        )
        for data in test_ds
    ]

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_data, batch_size=args.test_batch_size)
    evaluator(model, output_path=args.output_folder)


if __name__ == "__main__":
    args = parse(Args)
    main(args)
