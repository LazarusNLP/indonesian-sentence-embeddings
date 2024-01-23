from dataclasses import dataclass
import pickle
import random
import math
import os

from datargs import parse
from datasets import load_dataset
from sentence_transformers import InputExample, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers_congen import SentenceTransformer, losses
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn


@dataclass
class Args:
    # data args
    model_name: str = "indobenchmark/indobert-base-p1"
    # train
    train_dataset_name: str = "LazarusNLP/wikipedia_id_20230520"
    train_dataset_split: str = "train"
    train_text_column: str = "text"
    max_seq_length: int = 32
    max_train_samples: int = 1_000_000
    min_text_length: int = 20
    max_text_length: int = 200
    # test
    test_dataset_name: str = "LazarusNLP/stsb_mt_id"
    test_dataset_split: str = "validation"
    test_text_column_1: str = "text_1"
    test_text_column_2: str = "text_2"
    test_label_column: str = "correlation"
    # training args
    num_epochs: int = 20
    train_batch_size: int = 128
    test_batch_size: int = 32
    early_stopping_patience: int = 7
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    output_path: str = "exp/congen-indobert-base"
    # ConGen params
    teacher_model_name: str = (
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    queue_size: int = 65536
    student_temp: float = 0.5
    teacher_temp: float = 0.5
    # cached encoded datasets
    encoded_texts_path: str = "encoded_texts.pkl"
    instance_queue_encoded_path: str = "instance_queue_encoded.pkl"
    # huggingface hub args
    hub_model_id: str = "LazarusNLP/congen-indobert-base"
    hub_private_repo: bool = True


def corrupt(text: str) -> str:
    words = text.split()
    del_idx = random.choice(range(len(words)))
    return " ".join([w for i, w in enumerate(words) if i != del_idx])


def main(args: Args):
    # Load datasets
    train_ds = load_dataset(args.train_dataset_name, split=args.train_dataset_split)
    test_ds = load_dataset(args.test_dataset_name, split=args.test_dataset_split)

    # Preprocess train set
    num_proc = os.cpu_count()
    train_ds = train_ds.filter(
        lambda x: args.min_text_length < len(x["text"]) < args.max_text_length,
        num_proc=num_proc,
    )  # filter by length
    # select random train samples
    train_ds = train_ds.shuffle(seed=42).select(range(args.max_train_samples))

    # Initialize teacher model
    teacher_model = SentenceTransformer(args.teacher_model_name)

    # get teacher sentence embeddings
    try:
        encoded_texts = pickle.load(open(args.encoded_texts_path, "rb"))
    except:
        encoded_texts = teacher_model.encode(
            train_ds[args.train_text_column],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=128,
        )
        pickle.dump(encoded_texts, open(args.encoded_texts_path, "wb"), protocol=4)

    teacher_dimension = encoded_texts.shape[1]

    # Intialize student model with CLS pool and dense layer
    word_embedding_model = models.Transformer(
        args.model_name, max_seq_length=args.max_seq_length
    )
    dimension = word_embedding_model.get_word_embedding_dimension()
    pooling_model = models.Pooling(dimension)
    # project student's output pooling to teacher's output dimension
    dense_model = models.Dense(
        in_features=dimension,
        out_features=teacher_dimension,
        activation_function=nn.Tanh(),
    )
    model = SentenceTransformer(
        modules=[word_embedding_model, pooling_model, dense_model]
    )

    # create instance queue
    text_in_queue = np.random.RandomState(16349).choice(
        train_ds[args.train_text_column], args.queue_size, replace=False
    )

    train_samples, instance_queue = [], []
    text_in_q_set = set(text_in_queue)

    for sent, encoded_text in zip(train_ds[args.train_text_column], encoded_texts):
        # if sentence not in queue, add as training sample pairs
        if sent not in text_in_q_set:
            train_samples.append(
                InputExample(texts=[sent, corrupt(sent)], label=encoded_text)
            )
        # otherwise, add to queue
        else:
            instance_queue.append(sent)

    train_dataloader = DataLoader(
        train_samples, shuffle=True, batch_size=args.train_batch_size
    )

    try:
        instance_queue_encoded = pickle.load(
            open(args.instance_queue_encoded_path, "rb")
        )
    except:
        instance_queue_encoded = teacher_model.encode(
            instance_queue,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=128,
        )
        pickle.dump(
            instance_queue_encoded,
            open(args.instance_queue_encoded_path, "wb"),
            protocol=4,
        )

    # Use ConGen loss
    train_loss = losses.ConGenLoss(
        instanceQ_encoded=instance_queue_encoded,
        model=model,
        student_temp=args.student_temp,
        teacher_temp=args.teacher_temp,
    )

    del instance_queue, encoded_texts, teacher_model, instance_queue_encoded

    warmup_steps = math.ceil(
        len(train_dataloader) * args.num_epochs * args.warmup_ratio
    )  # 10% of train data for warm-up

    # Setup test data for evaluation
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

    # Call the fit method
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.num_epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        optimizer_params={"lr": args.learning_rate, "eps": 1e-6},
        output_path=args.output_path,
        save_best_model=True,
        early_stopping_patience=args.early_stopping_patience,
    )

    # Save model to HuggingFace Hub
    model.save_to_hub(
        args.hub_model_id,
        private=args.hub_private_repo,
        train_datasets=[args.train_dataset_name],
    )


if __name__ == "__main__":
    args = parse(Args)
    main(args)
