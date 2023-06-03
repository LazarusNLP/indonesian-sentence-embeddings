from dataclasses import dataclass
import os

from datargs import parse
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader


@dataclass
class Args:
    # data args
    model_name: str = "indobenchmark/indobert-base-p1"
    # train
    train_dataset_name: str = "LazarusNLP/wikipedia_id_20230520"
    train_dataset_split: str = "train"
    train_text_column_1: str = "text"
    train_text_column_2: str = "text"
    max_train_samples: int = 1_000_000
    max_seq_length: int = 32
    min_text_length: int = 20
    max_text_length: int = 200
    # test
    test_dataset_name: str = "LazarusNLP/stsb_mt_id"
    test_dataset_split: str = "validation"
    test_text_column_1: str = "text_1"
    test_text_column_2: str = "text_2"
    test_label_column: str = "correlation"
    # training args
    num_epochs: int = 1
    train_batch_size: int = 128
    test_batch_size: int = 32
    learning_rate: float = 3e-5
    output_path: str = "exp/simcse-indobert-base"
    # huggingface hub args
    hub_model_id: str = "LazarusNLP/simcse-indobert-base"
    hub_private_repo: bool = True


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

    # Intialize model with CLS pool
    word_embedding_model = models.Transformer(
        args.model_name, max_seq_length=args.max_seq_length
    )
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Convert train sentences to sentence pairs
    train_data = [
        InputExample(
            texts=[data[args.train_text_column_1], data[args.train_text_column_2]]
        )
        for data in train_ds
    ]

    # DataLoader to batch your data
    train_dataloader = DataLoader(
        train_data, batch_size=args.train_batch_size, shuffle=True
    )

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

    # Use the denoising auto-encoder loss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Call the fit method
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        epochs=args.num_epochs,
        show_progress_bar=True,
        optimizer_params={"lr": args.learning_rate},
        output_path=args.output_path,
        save_best_model=True,
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
