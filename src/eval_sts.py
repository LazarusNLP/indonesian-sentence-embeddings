from dataclasses import dataclass

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


@dataclass
class args:
    model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    test_dataset_name = "LazarusNLP/stsb_mt_id"
    test_dataset_split = "test"
    test_text_column_1 = "text_1"
    test_text_column_2 = "text_2"
    test_label_column = "correlation"
    test_batch_size = 32


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

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    test_data, batch_size=args.test_batch_size
)

print(evaluator(model))
