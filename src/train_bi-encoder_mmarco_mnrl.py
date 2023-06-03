from dataclasses import dataclass
import math

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader


@dataclass
class args:
    # data args
    model_name = "indobenchmark/indobert-base-p1"
    # train
    train_dataset_name = "unicamp-dl/mmarco"
    train_dataset_config = "indonesian"
    train_dataset_split = "train"
    train_query_column = "query"
    train_positive_column = "positive"
    train_negative_column = "negative"
    max_seq_length = 32
    # test
    test_dataset_name = "LazarusNLP/stsb_mt_id"
    test_dataset_split = "validation"
    test_text_column_1 = "text_1"
    test_text_column_2 = "text_2"
    test_label_column = "correlation"
    # training args
    num_epochs = 10
    train_batch_size = 64
    test_batch_size = 32
    learning_rate = 2e-5
    warmup_ratio = 0.1
    output_path = "exp/mmarco-indobert-base"
    # huggingface hub args
    hub_model_id = "LazarusNLP/mmarco-indobert-base"
    hub_private_repo = True


# Load datasets
train_ds = load_dataset(
    args.train_dataset_name, args.train_dataset_config, split=args.train_dataset_split
)
test_ds = load_dataset(args.test_dataset_name, split=args.test_dataset_split)

# Intialize model with mean pool
word_embedding_model = models.Transformer(
    args.model_name, max_seq_length=args.max_seq_length
)
dimension = word_embedding_model.get_word_embedding_dimension()
pooling_model = models.Pooling(dimension, pooling_mode="mean")
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Convert train sentences to query-positive-negative triplets
train_data = [
    InputExample(
        texts=[
            data[args.train_query_column],
            data[args.train_positive_column],
            data[args.train_negative_column],
        ]
    )
    for data in train_ds
]

# DataLoader to batch your data
train_dataloader = DataLoader(
    train_data, batch_size=args.train_batch_size, shuffle=True
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
    warmup_steps=warmup_steps,
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
