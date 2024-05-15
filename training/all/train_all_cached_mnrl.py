from dataclasses import dataclass
import math

from datargs import parse
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from all_datasets import (
    IndoNLI,
    IndoStoryCloze,
    mMARCO,
    MIRACL,
    SwimIR,
    MultilingualNLI,
    WReTE,
    IndoLEMNTP,
    TyDiQA,
    FacQA,
    LFQAID,
    IndoQA,
    ParaphraseDetection,
)
from MultiDatasetDataLoader import MultiDatasetDataLoader


@dataclass
class Args:
    # data args
    model_name: str = "indobenchmark/indobert-base-p1"
    # train
    max_seq_length: int = 128
    # test
    test_dataset_name: str = "LazarusNLP/stsb_mt_id"
    test_dataset_split: str = "validation"
    test_text_column_1: str = "text_1"
    test_text_column_2: str = "text_2"
    test_label_column: str = "correlation"
    # training args
    num_epochs: int = 5
    train_batch_size_pairs: int = 384
    train_batch_size_triplets: int = 256
    test_batch_size: int = 32
    mini_batch_size: int = 128
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
        "indonli": IndoNLI,
        "indolem/indo_story_cloze": IndoStoryCloze,
        "unicamp-dl/mmarco": mMARCO,
        "miracl/miracl": MIRACL,
        "nthakur/swim-ir-monolingual": SwimIR,
        "LazarusNLP/multilingual-NLI-26lang-2mil7-id": MultilingualNLI,
        "SEACrowd/wrete": WReTE,
        "SEACrowd/indolem_ntp": IndoLEMNTP,
        "khalidalt/tydiqa-goldp": TyDiQA,
        "SEACrowd/facqa": FacQA,
        "indonesian-nlp/lfqa_id": LFQAID,
        "jakartaresearch/indoqa": IndoQA,
        "jakartaresearch/id-paraphrase-detection": ParaphraseDetection,
    }

    train_ds = [ds.train_samples() for ds in raw_datasets.values()]
    test_ds = load_dataset(args.test_dataset_name, split=args.test_dataset_split)

    # Intialize model with mean pool
    word_embedding_model = models.Transformer(args.model_name, max_seq_length=args.max_seq_length)
    dimension = word_embedding_model.get_word_embedding_dimension()
    pooling_model = models.Pooling(dimension, pooling_mode="mean")
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # DataLoader to batch your data
    train_dataloader = MultiDatasetDataLoader(
        train_ds, batch_size_pairs=args.train_batch_size_pairs, batch_size_triplets=args.train_batch_size_triplets
    )

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

    # Use the denoising auto-encoder loss
    train_loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size)

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
        use_amp=args.use_amp,
    )

    # Save model to HuggingFace Hub
    model.save_to_hub(
        args.hub_model_id,
        private=args.hub_private_repo,
        train_datasets=list(raw_datasets.keys()),
    )


if __name__ == "__main__":
    args = parse(Args)
    main(args)
