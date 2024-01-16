# Evaluation

## Machine Translated STS-B

To the best of our knowledge, there is no official benchmark on Indonesian sentence embeddings. Inspired by [Thai Sentence Vector Benchmark](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark), we translated the [STS-B](https://github.com/facebookresearch/SentEval) dev and test set to Indonesian via Google Translate API. This dataset will be used to evaluate our model's Spearman correlation score on the translated test set. You can find the translated dataset on [🤗 HuggingFace Hub](https://huggingface.co/datasets/LazarusNLP/stsb_mt_id).

For practical purposes, we used Sentence Transformer's [`EmbeddingSimilarityEvaluator`](https://www.sbert.net/docs/package_reference/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator) to perform inference and evaluate our models.

### Example

```sh
python eval_sts.py \
    --model-name LazarusNLP/congen-indobert-base \
    --test-dataset-name LazarusNLP/stsb_mt_id \
    --test-dataset-split test \
    --test-text-column-1 text_1 \
    --test-text-column-2 text_2 \
    --test-label-column correlation \
    --test-batch-size 32
```

## References

```bibtex
@misc{Thai-Sentence-Vector-Benchmark-2022,
  author = {Limkonchotiwat, Peerat},
  title = {Thai-Sentence-Vector-Benchmark},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark}}
}
```

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