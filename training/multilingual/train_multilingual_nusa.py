from dataclasses import dataclass
from itertools import chain
import math

from datargs import parse
from datasets import load_dataset
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, models, losses
from sentence_transformers.datasets import ParallelSentencesDataset
from sentence_transformers.evaluation import (
    MSEEvaluator,
    TranslationEvaluator,
    EmbeddingSimilarityEvaluator,
    SequentialEvaluator,
)

import numpy as np
import torch.nn as nn

from all_datasets import NusaX, NusaTranslation


@dataclass
class Args:
    # data args
    student_model_name: str = "LazarusNLP/NusaBERT-base"
    teacher_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    # train
    max_seq_length: int = 128
    # test
    test_dataset_name: str = "LazarusNLP/stsb_mt_id"
    test_dataset_split: str = "validation"
    test_text_column_1: str = "text_1"
    test_text_column_2: str = "text_2"
    test_label_column: str = "correlation"
    # training args
    num_epochs: int = 20
    train_batch_size: int = 128
    test_batch_size: int = 128
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    output_path: str = "exp/all-indobert-base"
    use_amp: bool = True
    # huggingface hub args
    hub_model_id: str = "LazarusNLP/all-indobert-base"
    hub_private_repo: bool = True


def main(args: Args):
    # Load datasets
    raw_datasets = {
        "indonlp/NusaX-MT": NusaX,
        "indonlp/nusatranslation_mt": NusaTranslation,
    }

    train_ds = [ds.train_samples() for ds in raw_datasets.values()]
    train_ds = list(chain.from_iterable(train_ds))  # flatten multiple datasets

    dev_ds = [ds.validation_samples() for ds in raw_datasets.values()]
    dev_ds = list(chain.from_iterable(dev_ds))

    test_ds = load_dataset(args.test_dataset_name, split=args.test_dataset_split)

    # Load teacher model
    teacher_model = SentenceTransformer(args.teacher_model_name)
    teacher_dimension = teacher_model.get_sentence_embedding_dimension()

    # Intialize model with mean pool
    word_embedding_model = models.Transformer(args.student_model_name, max_seq_length=args.max_seq_length)
    dimension = word_embedding_model.get_word_embedding_dimension()
    pooling_model = models.Pooling(dimension, pooling_mode="mean")
    # project student's output pooling to teacher's output dimension
    dense_model = models.Dense(
        in_features=dimension,
        out_features=teacher_dimension,
        activation_function=nn.Tanh(),
    )
    student_model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

    # Prepare Parallel Sentences Dataset
    parallel_ds = ParallelSentencesDataset(
        student_model=student_model,
        teacher_model=teacher_model,
        batch_size=args.test_batch_size,
        use_embedding_cache=True,
    )
    parallel_ds.add_dataset(train_ds, max_sentence_length=args.max_seq_length)

    # DataLoader to batch your data
    train_dataloader = DataLoader(parallel_ds, batch_size=args.train_batch_size)

    warmup_steps = math.ceil(
        len(train_dataloader) * args.num_epochs * args.warmup_ratio
    )  # 10% of train data for warm-up

    # Flatten validation translation pairs into two separate lists
    source_sentences, target_sentences = map(list, zip(*dev_ds))

    # MSE evaluation
    mse_evaluator = MSEEvaluator(
        source_sentences, target_sentences, teacher_model=teacher_model, batch_size=args.test_batch_size
    )

    # Translation evaluation
    trans_evaluator = TranslationEvaluator(source_sentences, target_sentences, batch_size=args.test_batch_size)

    # STS evaluation
    test_data = [
        InputExample(
            texts=[data[args.test_text_column_1], data[args.test_text_column_2]],
            label=float(data[args.test_label_column]) / 5.0,
        )
        for data in test_ds
    ]

    sts_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_data, batch_size=args.test_batch_size)

    # Use MSE crosslingual distillation loss
    train_loss = losses.MSELoss(model=student_model)

    # Call the fit method
    student_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=SequentialEvaluator(
            [mse_evaluator, trans_evaluator, sts_evaluator], main_score_function=lambda scores: np.mean(scores)
        ),
        epochs=args.num_epochs,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        optimizer_params={"lr": args.learning_rate, "eps": 1e-6},
        output_path=args.output_path,
        save_best_model=True,
        use_amp=args.use_amp,
    )

    # Save model to HuggingFace Hub
    student_model.save_to_hub(
        args.hub_model_id,
        private=args.hub_private_repo,
        train_datasets=list(raw_datasets.keys()),
    )


if __name__ == "__main__":
    args = parse(Args)
    main(args)
