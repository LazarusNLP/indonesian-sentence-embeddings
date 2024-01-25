# All Supervised Datasets

Inspired by [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), we fine-tuned Indonesian sentence embedding models on a set of existing supervised datasets. The tasks included in the training dataset are: question-answering, textual entailment, retrieval, commonsense reasoning, and natural language inference. Currently, our script simply concatenates these datasets and our models are trained conventionally using the `MultipleNegativesRankingLoss`.

## Training Data

| Dataset   |            Task            |   Type   | Number of Training Tuples |
| --------- | :------------------------: | :------: | :-----------------------: |
| indonli   | Natural Language Inference | triplets |           3,914           |
|           |                            |          |                           |
|           |                            |          |                           |
|           |                            |          |                           |
|           |                            |          |                           |
|           |                            |          |                           |
|           |                            |          |                           |
|           |                            |          |                           |
| **Total** |                            |          |        **135,258**        |

## All Supervised Datasets with MultipleNegativesRankingLoss

### IndoBERT Base

```sh
python train_all_mnrl.py \
    --model-name indobenchmark/indobert-base-p1 \
    --max-seq-length 128 \
    --num-epochs 5 \
    --train-batch-size-pairs 384 \
    --train-batch-size-triplets 256 \
    --learning-rate 2e-5
```

## References

```bibtex
@inproceedings{reimers-2019-sentence-bert,
  title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
  author = "Reimers, Nils and Gurevych, Iryna",
  booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
  month = "11",
  year = "2019",
  publisher = "Association for Computational Linguistics",
  url = "https://arxiv.org/abs/1908.10084",
}
```