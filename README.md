# Indonesian Sentence Embeddings

Inspired by [Thai Sentence Vector Benchmark](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark), we decided to embark on the journey of training Indonesian sentence embedding models!

To the best of our knowledge, there is no official benchmark on Indonesian sentence embeddings. We hope this repository can serve as a benchmark for future research on Indonesian sentence embeddings.

## Evaluation

### Machine Translated STS-B

We believe that a synthetic baseline is better than no baseline. Therefore, we followed approached done in the Thai Sentence Vector Benchmark project and translated the [STS-B](https://github.com/facebookresearch/SentEval) dev and test set to Indonesian via Google Translate API. This dataset will be used to evaluate our model's Spearman correlation score on the translated test set.

> You can find the translated dataset on [🤗 HuggingFace Hub](https://huggingface.co/datasets/LazarusNLP/stsb_mt_id).

Moreover, we will further evaluate the transferrability of our models on downstream tasks (e.g. text classification, natural language inference, etc.) and compare them with existing pre-trained language models (PLMs).

### Text Classification

For text classification, we will be doing emotion classification and sentiment analysis on the EmoT and SmSA subsets of [IndoNLU](https://huggingface.co/datasets/indonlp/indonlu), respectively. To do so, we will be doing the same approach as Thai Sentence Vector Benchmark and simply fit a Linear SVC on sentence representations of our texts with their corresponding labels. Thus, unlike conventional fine-tuning method where the backbone model is also updated, the Sentence Transformer stays frozen in our case; with only the classification head being trained.

## Methods

### (Unsupervised) SimCSE

We followed [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821) and trained a sentence embedding model in an unsupervised fashion. Unsupervised SimCSE allows us to leverage an unsupervised corpus -- which are plenty -- and with different dropout masks in the encoder, contrastively learn sentence representations. This is parallel with the situation that there is a lack of supervised Indonesian sentence similarity datasets, hence SimCSE is a natural first move into this field. We used the [Sentence Transformer implementation](https://www.sbert.net/examples/unsupervised_learning/README.html#simcse) of [SimCSE](https://github.com/princeton-nlp/SimCSE).

### ConGen

Like SimCSE, [ConGen: Unsupervised Control and Generalization Distillation For Sentence Representation](https://github.com/KornWtp/ConGen) is another unsupervised technique to train a sentence embedding model. Since it is in-part a distillation method, ConGen relies on a teacher model which will then be distilled to a student model. The original paper proposes back-translation as the best data augmentation technique. However, due to the lack of resources, we implemented word deletion, which was found to be on-par with back-translation despite being trivial. We used the [official ConGen implementation](https://github.com/KornWtp/ConGen) which was written on top of the Sentence Transformers library.

## Results

### Semantic Textual Similarity

| Model                                                                                                                       | Spearman's Correlation (%) | #params | Base Model                                                                                | Train Dataset                                                                     |
| --------------------------------------------------------------------------------------------------------------------------- | :------------------------: | :-----: | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| [SimCSE-IndoBERT Base](https://huggingface.co/LazarusNLP/simcse-indobert-base)                                              |           70.13            |  125M   | [IndoBERT Base](https://huggingface.co/indobenchmark/indobert-base-p1)                    | [Wikipedia](https://huggingface.co/datasets/LazarusNLP/wikipedia_id_20230520)     |
| [SimCSE-IndoRoBERTa Base](https://huggingface.co/LazarusNLP/simcse-indoroberta-base)                                        |           61.26            |  125M   | [IndoRoBERTa Base](https://huggingface.co/flax-community/indonesian-roberta-base)         | [Wikipedia](https://huggingface.co/datasets/LazarusNLP/wikipedia_id_20230520)     |
| [ConGen-IndoBERT Base](https://huggingface.co/LazarusNLP/congen-indobert-base)                                              |           80.47            |  125M   | [IndoBERT Base](https://huggingface.co/indobenchmark/indobert-base-p1)                    | [Wikipedia](https://huggingface.co/datasets/LazarusNLP/wikipedia_id_20230520)     |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)   |           75.08            |  134M   | [DistilBERT Base Multilingual](https://huggingface.co/distilbert-base-multilingual-cased) | See [SBERT.net](https://www.sbert.net/docs/pretrained_models.html#model-overview) |
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) |           83.83            |  125M   | [XLM-RoBERTa Base](https://huggingface.co/xlm-roberta-base)                               | See [SBERT.net](https://www.sbert.net/docs/pretrained_models.html#model-overview) |

### Emotion Classification (EmoT)

| Model                                                                                                                       | Accuracy (%) | F1 Macro (%) |
| --------------------------------------------------------------------------------------------------------------------------- | :----------: | :----------: |
| [SimCSE-IndoBERT Base](https://huggingface.co/LazarusNLP/simcse-indobert-base)                                              |    55.90     |    56.34     |
| [SimCSE-IndoRoBERTa Base](https://huggingface.co/LazarusNLP/simcse-indoroberta-base)                                        |    50.45     |    50.40     |
| [ConGen-IndoBERT Base](https://huggingface.co/LazarusNLP/congen-indobert-base)                                              |    56.81     |    56.94     |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)   |    63.63     |    64.13     |
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) |    63.18     |    63.78     |

### Sentiment Analysis (SmSA)

| Model                                                                                                                       | Accuracy (%) | F1 Macro (%) |
| --------------------------------------------------------------------------------------------------------------------------- | :----------: | :----------: |
| [SimCSE-IndoBERT Base](https://huggingface.co/LazarusNLP/simcse-indobert-base)                                              |     85.0     |    81.00     |
| [SimCSE-IndoRoBERTa Base](https://huggingface.co/LazarusNLP/simcse-indoroberta-base)                                        |     75.8     |    69.90     |
| [ConGen-IndoBERT Base](https://huggingface.co/LazarusNLP/congen-indobert-base)                                              |     85.4     |    82.12     |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)   |     78.8     |    73.64     |
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) |     89.6     |    86.56     |

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

```bibtex
@inproceedings{gao2021simcse,
  title={{SimCSE}: Simple Contrastive Learning of Sentence Embeddings},
  author={Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
  booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
  year={2021}
}
```