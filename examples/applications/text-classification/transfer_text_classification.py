# Modified from: https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/Transfer_Evaluation/transfer.py

from dataclasses import dataclass

from datargs import parse
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


@dataclass
class Args:
    model_name: str = "LazarusNLP/simcse-indobert-base"
    dataset_name: str = "indonlp/indonlu"
    dataset_config: str = "emot"
    train_split_name: str = "train"
    test_split_name: str = "test"
    text_column: str = "tweet"
    label_column: str = "label"
    encode_batch_size: int = 128


def main(args: Args):
    model = SentenceTransformer(args.model_name)

    dataset = load_dataset(args.dataset_name, args.dataset_config)
    train_ds, test_ds = dataset[args.train_split_name], dataset[args.test_split_name]

    # encode sentence embeddings
    train_text_encoded = model.encode(
        train_ds[args.text_column],
        batch_size=args.encode_batch_size,
        show_progress_bar=True,
    )

    test_text_encoded = model.encode(
        test_ds[args.text_column],
        batch_size=args.encode_batch_size,
        show_progress_bar=True,
    )

    classifier = LinearSVC()
    classifier.fit(train_text_encoded, train_ds[args.label_column])

    predictions = classifier.predict(test_text_encoded)

    acc = accuracy_score(test_ds[args.label_column], predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_ds[args.label_column], predictions, average="macro"
    )

    results = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    print(results)


if __name__ == "__main__":
    args = parse(Args)
    main(args)
