from dataclasses import dataclass
import pickle
import random
import math
import os

from datargs import parse
from datasets import load_dataset
from sentence_transformers import InputExample, models, SentenceTransformer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class Args:
    # data args
    model_name: str = "indobenchmark/indobert-base-p1"
    # train
    train_dataset_name: str = "LazarusNLP/wikipedia_id_backtranslated"
    train_dataset_split: str = "train"
    train_text_column_1: str = "text"
    train_text_column_2: str = "text_bt"
    max_seq_length: int = 128
    max_train_samples: int = 1_000_000
    min_text_length: int = 150
    max_text_length: int = 500
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
    output_path: str = "exp/sct-indobert-base"
    # SCT params
    teacher_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    queue_size: int = 65536
    student_temp: float = 0.5
    teacher_temp: float = 0.5
    do_corrupt: bool = False
    # cached encoded datasets
    encoded_ref_1_path: str = "encoded_ref_1.pkl"
    encoded_ref_2_path: str = "encoded_ref_2.pkl"
    # huggingface hub args
    hub_model_id: str = "LazarusNLP/sct-indobert-base"
    hub_private_repo: bool = True


def corrupt(text: str) -> str:
    words = text.split()
    del_idx = random.choice(range(len(words)))
    return " ".join([w for i, w in enumerate(words) if i != del_idx])


def main(args: Args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        encoded_ref_1 = pickle.load(open(args.encoded_ref_1_path, "rb"))
    except:
        encoded_ref_1 = teacher_model.encode(
            train_ds[args.train_text_column_1],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            device=device,
            batch_size=512,
        )
        pickle.dump(encoded_ref_1, open(args.encoded_ref_1_path, "wb"), protocol=4)

    try:
        encoded_ref_2 = pickle.load(open(args.encoded_ref_2_path, "rb"))
    except:
        if args.do_corrupt:
            sentences = [corrupt(text) for text in train_ds[args.train_text_column_1]]
        else:
            sentences = train_ds[args.train_text_column_2]

        encoded_ref_2 = teacher_model.encode(
            sentences,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
            device=device,
            batch_size=512,
        )
        pickle.dump(encoded_ref_2, open(args.encoded_ref_2_path, "wb"), protocol=4)

    teacher_dimension = encoded_ref_1.shape[1]

    # Intialize student model with CLS pool and dense layer
    word_embedding_model = models.Transformer(args.model_name, max_seq_length=args.max_seq_length)
    dimension = word_embedding_model.get_word_embedding_dimension()
    pooling_model = models.Pooling(dimension)
    # project student's output pooling to teacher's output dimension
    if teacher_dimension != dimension:
        dense_model = models.Dense(in_features=dimension, out_features=teacher_dimension, activation_function=nn.Tanh())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
    else:
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # create instance queues
    rep_instance_queue_edited_A = torch.randn(args.queue_size, teacher_dimension).to(device)
    rep_instance_queue_edited_A = F.normalize(rep_instance_queue_edited_A, p=2, dim=1)

    rep_instance_queue_edited_B = torch.randn(args.queue_size, teacher_dimension).to(device)
    rep_instance_queue_edited_B = F.normalize(rep_instance_queue_edited_B, p=2, dim=1)

    # prepare dataloaders
    train_samples = []
    for x1, x2, ref_1, ref_2 in zip(
        train_ds[args.train_text_column_1], train_ds[args.train_text_column_2], encoded_ref_1, encoded_ref_2
    ):
        train_samples.append(InputExample(texts=[x1, x2], label=[ref_1, ref_2]))

    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.train_batch_size)

    # Use SCT distillation loss
    train_loss = losses.SCTLoss_distillation(
        instanceQ_A=rep_instance_queue_edited_A,
        instanceQ_B=rep_instance_queue_edited_B,
        model=model,
        student_temp=args.student_temp,
        teacher_temp=args.teacher_temp,
        device=device,
        sentence_embedding_dimension=teacher_dimension,
    )

    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * args.warmup_ratio)

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
        use_amp=True,
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
