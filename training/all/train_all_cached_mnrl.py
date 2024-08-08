from dataclasses import dataclass

from datargs import parse
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import (
    BatchSamplers,
    MultiDatasetBatchSamplers,
    SentenceTransformerTrainingArguments,
)

from all_datasets import *


@dataclass
class Args:
    # data args
    model_name: str = "indobenchmark/indobert-base-p1"
    # train
    max_seq_length: int = 128
    # eval
    eval_dataset_name: str = "LazarusNLP/stsb_mt_id"
    eval_dataset_split: str = "validation"
    eval_text_column_1: str = "text_1"
    eval_text_column_2: str = "text_2"
    eval_label_column: str = "correlation"
    # training args
    num_epochs: int = 5
    train_batch_size: int = 256
    eval_batch_size: int = 32
    mini_batch_size: int = 128
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    output_path: str = "exp/all-indobert-base"
    bf16: bool = True
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
        "LazarusNLP/multilingual-NLI-26lang-2mil7-id": MultilingualNLIEntailmentPairs,
        "SEACrowd/wrete": WReTEEntailmentPairs,
        "SEACrowd/indolem_ntp": IndoLEMNTPEntailmentPairs,
        "khalidalt/tydiqa-goldp": TyDiQA,
        "SEACrowd/facqa": FacQA,
        "indonesian-nlp/lfqa_id": LFQAID,
        "jakartaresearch/indoqa": IndoQA,
        "jakartaresearch/id-paraphrase-detection": ParaphraseDetection,
        "wikimedia/wikipedia": Wikipedia,
        "lesserfield/brainly": Brainly,
        "esteler-ai/idn-news-az": IndonesianNews,
        "hermanshid/doctor-id-qa": DoctorQA,
        "SEACrowd/liputan6": Liputan6,
    }

    train_dataset = {name: ds().dataset for name, ds in raw_datasets.items()}

    eval_dataset = load_dataset(args.eval_dataset_name, split=args.eval_dataset_split)
    eval_dataset = eval_dataset.map(lambda score: {"score": float(score) / 5.0}, input_columns=[args.eval_label_column])
    eval_dataset = eval_dataset.select_columns([args.eval_text_column_1, args.eval_text_column_2, "score"])
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=eval_dataset[args.eval_text_column_1],
        sentences2=eval_dataset[args.eval_text_column_2],
        scores=eval_dataset["score"],
        main_similarity=SimilarityFunction.COSINE,
        name="sts-dev",
    )

    # Intialize model with mean pool
    word_embedding_model = models.Transformer(args.model_name, max_seq_length=args.max_seq_length)
    dimension = word_embedding_model.get_word_embedding_dimension()
    pooling_model = models.Pooling(dimension, pooling_mode="mean")
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    mnrl_loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size)

    args = SentenceTransformerTrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        push_to_hub=True,
        hub_model_id=args.hub_model_id,
        hub_private_repo=args.hub_private_repo,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=mnrl_loss,
        evaluator=evaluator,
    )

    trainer.train()

    trainer.push_to_hub(dataset=list(raw_datasets.keys()))


if __name__ == "__main__":
    args = parse(Args)
    main(args)
