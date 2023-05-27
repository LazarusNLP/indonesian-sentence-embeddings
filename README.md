# Indonesian Sentence Embeddings

Inspired by [Thai Sentence Vector Benchmark](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark), we decided to embark on the journey of training Indonesian sentence embedding models!

To the best of our knowledge, there is no official benchmark on Indonesian sentence embeddings. We hope this repository can serve as a benchmark for future research on Indonesian sentence embeddings.

## Machine Translated STS-B

We believe that a synthetic baseline is better than no baseline. Therefore, we followed approached done in the Thai Sentence Vector Benchmark project and translated the [STS-B](https://github.com/facebookresearch/SentEval) test set to Indonesian via Google Translate API. This dataset will be used to evaluate our model's Spearman correlation score on the translated test set.

> You can find the translated dataset on [🤗 HuggingFace Hub](https://huggingface.co/datasets/LazarusNLP/stsb_mt_id).

Moreover, we will further evaluate the transferrability of our models on downstream tasks (e.g. text classification, natural language inference, etc.) and compare them with existing pre-trained language models (PLMs).

## Methods

### (Unsupervised) SimCSE

We followed [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821) and trained a sentence embedding model in an unsupervised fashion. Unsupervised SimCSE allows us to leverage an unsupervised corpus -- which are plenty -- and with different dropout masks in the encoder, contrastively learn sentence representations. This is parallel with the situation that there is a lack of supervised Indonesian sentence similarity datasets, hence SimCSE is a natural first move into this field. We used the [Sentence Transformer implementation](https://www.sbert.net/examples/unsupervised_learning/README.html#simcse) of [SimCSE](https://github.com/princeton-nlp/SimCSE).

## Results

### Semantic Textual Similarity

| Model                                                                                | Spearman's Correlation (%) | Base Model                                                             | Train Dataset                                                                 |
| ------------------------------------------------------------------------------------ | :------------------------: | ---------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [SimCSE-IndoBERT Base](https://huggingface.co/LazarusNLP/simcse-indobert-base)       |           61.54%           | [IndoBERT Base](https://huggingface.co/indobenchmark/indobert-base-p1) | [Wikipedia](https://huggingface.co/datasets/LazarusNLP/wikipedia_id_20230520) |
| [SimCSE-IndoRoBERTa Base](https://huggingface.co/LazarusNLP/simcse-indoroberta-base) |           52.62%           | [IndoRoBERTa Base](flax-community/indonesian-roberta-base)             | [Wikipedia](https://huggingface.co/datasets/LazarusNLP/wikipedia_id_20230520) |

<!-- TODO: SmSA -->

<!-- TODO: EmoT -->

## References

```bibtex
@misc{Thai-Sentence-Vector-Benchmark-2022,
  author = {Mr.P L},
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

```bibtex
@inproceedings{gao2021simcse,
   title={{SimCSE}: Simple Contrastive Learning of Sentence Embeddings},
   author={Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2021}
}
```