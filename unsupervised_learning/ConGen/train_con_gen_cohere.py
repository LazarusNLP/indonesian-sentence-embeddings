from dataclasses import dataclass
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
import torch
import torch.nn as nn


@dataclass
class Args:
    # data args
    model_name: str = "LazarusNLP/NusaBERT-base"
    # train
    train_dataset_name: str = "Cohere/wikipedia-2023-11-embed-multilingual-v3"
    train_dataset_config: str = "id"
    train_dataset_split: str = "train"
    train_text_column: str = "text"
    train_embeddings_column: str = "emb"
    max_seq_length: int = 128
    max_train_samples: int = 1_000_000
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
    output_path: str = "exp/congen-nusabert-base"
    # ConGen params
    queue_size: int = 65536
    student_temp: float = 0.5
    teacher_temp: float = 0.5
    # huggingface hub args
    hub_model_id: str = "LazarusNLP/congen-nusabert-base"
    hub_private_repo: bool = True


def corrupt(text: str) -> str:
    words = text.split()
    del_idx = random.choice(range(len(words)))
    return " ".join([w for i, w in enumerate(words) if i != del_idx])


def main(args: Args):
    # Load datasets
    train_ds = load_dataset(args.train_dataset_name, args.train_dataset_config, split=args.train_dataset_split)
    train_ds = train_ds.with_format(type="torch", columns=[args.train_embeddings_column])
    test_ds = load_dataset(args.test_dataset_name, split=args.test_dataset_split)

    # select random train samples
    train_ds = train_ds.shuffle(seed=42).select(range(args.max_train_samples))

    # get teacher sentence embeddings
    encoded_texts = train_ds[args.train_embeddings_column]
    teacher_dimension = encoded_texts.shape[1]

    # Intialize student model with CLS pool and dense layer
    word_embedding_model = models.Transformer(args.model_name, max_seq_length=args.max_seq_length)
    dimension = word_embedding_model.get_word_embedding_dimension()
    pooling_model = models.Pooling(dimension)
    # project student's output pooling to teacher's output dimension
    dense_model = models.Dense(
        in_features=dimension,
        out_features=teacher_dimension,
        activation_function=nn.Tanh(),
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

    # move encoded texts to the same device as model
    encoded_texts = encoded_texts.to(model._target_device)

    # create instance queue
    text_in_queue = np.random.RandomState(16349).choice(
        train_ds[args.train_text_column], args.queue_size, replace=False
    )

    train_samples, instance_queue_encoded = [], []
    text_in_q_set = set(text_in_queue)

    for sent, encoded_text in zip(train_ds[args.train_text_column], encoded_texts):
        # if sentence not in queue, add as training sample pairs
        if sent not in text_in_q_set:
            train_samples.append(InputExample(texts=[sent, corrupt(sent)], label=encoded_text))
        # otherwise, add to queue
        else:
            instance_queue_encoded.append(encoded_text)

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size)

    # convert list of 1d embeddings tensor as a 2d tensor
    instance_queue_encoded = torch.stack(instance_queue_encoded)

    # Use ConGen loss
    train_loss = losses.ConGenLoss(
        instanceQ_encoded=instance_queue_encoded,
        model=model,
        student_temp=args.student_temp,
        teacher_temp=args.teacher_temp,
    )

    del encoded_texts, instance_queue_encoded

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

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_data, batch_size=args.test_batch_size)

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
