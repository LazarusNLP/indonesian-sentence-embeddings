# Modified from: https://github.com/embeddings-benchmark/mteb/blob/main/mteb/evaluation/evaluators/PairClassificationEvaluator.py

from dataclasses import dataclass

import numpy as np
from datargs import parse
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import average_precision_score
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances


@dataclass
class Args:
    model_name: str = "LazarusNLP/simcse-indobert-base"
    dataset_name: str = "indonli"
    test_split_name: str = "test_lay"
    text_column_1: str = "premise"
    text_column_2: str = "hypothesis"
    label_column: str = "label"
    entailment_label: int = 0
    neutral_label: int = 1
    contradiction_label: int = 2
    encode_batch_size: int = 128


def compute_metrics(model, sentences_1, sentences_2, labels, batch_size):
    sentences = list(set(sentences_1 + sentences_2))
    embeddings = np.asarray(model.encode(sentences, batch_size=batch_size))
    emb_dict = {sent: emb for sent, emb in zip(sentences, embeddings)}
    embeddings1 = [emb_dict[sent] for sent in sentences_1]
    embeddings2 = [emb_dict[sent] for sent in sentences_2]

    cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
    manhattan_distances = paired_manhattan_distances(embeddings1, embeddings2)
    euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)

    embeddings1_np = np.asarray(embeddings1)
    embeddings2_np = np.asarray(embeddings2)
    dot_scores = [np.dot(embeddings1_np[i], embeddings2_np[i]) for i in range(len(embeddings1_np))]

    labels = np.asarray(labels)
    output_scores = {}
    for short_name, name, scores, reverse in [
        ["cos_sim", "Cosine-Similarity", cosine_scores, True],
        ["manhattan", "Manhattan-Distance", manhattan_distances, False],
        ["euclidean", "Euclidean-Distance", euclidean_distances, False],
        ["dot", "Dot-Product", dot_scores, True],
    ]:
        output_scores[short_name] = _compute_metrics(scores, labels, reverse)

    return output_scores


def _compute_metrics(scores, labels, high_score_more_similar):
    acc, acc_threshold = find_best_acc_and_threshold(scores, labels, high_score_more_similar)
    f1, precision, recall, f1_threshold = find_best_f1_and_threshold(scores, labels, high_score_more_similar)
    ap = ap_score(scores, labels, high_score_more_similar)

    return {
        "accuracy": acc,
        "accuracy_threshold": acc_threshold,
        "f1": f1,
        "f1_threshold": f1_threshold,
        "precision": precision,
        "recall": recall,
        "ap": ap,
    }


def find_best_acc_and_threshold(scores, labels, high_score_more_similar: bool):
    assert len(scores) == len(labels)
    rows = list(zip(scores, labels))

    rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

    max_acc = 0
    best_threshold = -1

    positive_so_far = 0
    remaining_negatives = sum(np.array(labels) == 0)

    for i in range(len(rows) - 1):
        score, label = rows[i]
        if label == 1:
            positive_so_far += 1
        else:
            remaining_negatives -= 1

        acc = (positive_so_far + remaining_negatives) / len(labels)
        if acc > max_acc:
            max_acc = acc
            best_threshold = (rows[i][0] + rows[i + 1][0]) / 2

    return max_acc, best_threshold


def find_best_f1_and_threshold(scores, labels, high_score_more_similar: bool):
    assert len(scores) == len(labels)

    scores = np.asarray(scores)
    labels = np.asarray(labels)

    rows = list(zip(scores, labels))

    rows = sorted(rows, key=lambda x: x[0], reverse=high_score_more_similar)

    best_f1 = best_precision = best_recall = 0
    threshold = 0
    nextract = 0
    ncorrect = 0
    total_num_duplicates = sum(labels)

    for i in range(len(rows) - 1):
        score, label = rows[i]
        nextract += 1

        if label == 1:
            ncorrect += 1

        if ncorrect > 0:
            precision = ncorrect / nextract
            recall = ncorrect / total_num_duplicates
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                threshold = (rows[i][0] + rows[i + 1][0]) / 2

    return best_f1, best_precision, best_recall, threshold


def ap_score(scores, labels, high_score_more_similar: bool):
    return average_precision_score(labels, scores * (1 if high_score_more_similar else -1))


def main(args: Args):
    model = SentenceTransformer(args.model_name)

    test_ds = load_dataset(args.dataset_name, split=args.test_split_name, trust_remote_code=True)

    # Remove neutral pairs
    test_ds = test_ds.filter(lambda ex: ex[args.label_column] != args.neutral_label)
    # Re-map contradiction label to 0; entailment label to 1
    remap_labels = {args.contradiction_label: 0, args.entailment_label: 1}
    test_ds = test_ds.map(lambda ex: {"label": remap_labels[ex[args.label_column]]})

    scores = compute_metrics(
        model,
        sentences_1=test_ds[args.text_column_1],
        sentences_2=test_ds[args.text_column_2],
        labels=test_ds["label"],
        batch_size=args.encode_batch_size,
    )
    # Main score is the max of Average Precision (AP)
    main_score = max(scores[short_name]["ap"] for short_name in scores)
    scores["main_score"] = main_score

    print(scores)


if __name__ == "__main__":
    args = parse(Args)
    main(args)
