# Modified from: https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark/blob/main/Transfer_Evaluation/transfer.py

from dataclasses import dataclass

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


@dataclass
class args:
    model_name = "LazarusNLP/simcse-indobert-base"
    dataset_name = "indonlp/indonlu"
    dataset_config = "emot"
    train_split_name = "train"
    test_split_name = "test"
    text_column = "tweet"
    label_column = "label"


def preprocess(batch):
    sentences = batch[args.text_column]
    batch[args.text_column] = model.encode(sentences)
    return batch


model = SentenceTransformer(args.model_name)

dataset = load_dataset(args.dataset_name, args.dataset_config)
dataset = dataset.map(preprocess)
train_ds, test_ds = dataset[args.train_split_name], dataset[args.test_split_name]

classifier = LinearSVC()
classifier.fit(train_ds[args.text_column], train_ds[args.label_column])

predictions = classifier.predict(test_ds[args.text_column])

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