# Text Classification

Normally, sentence embedding models are leveraged for downstream tasks such as information retrieval, semantic search, clustering, etc. Text classification could similarly leverage these models' sentence embedding capabilities.

We will be doing emotion classification and sentiment analysis on the EmoT and SmSA subsets of [IndoNLU](https://huggingface.co/datasets/indonlp/indonlu), respectively. To do so, we will be doing the same approach as [Thai Sentence Vector Benchmark](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark) and simply fit a Linear SVC on sentence representations of our texts with their corresponding labels. Thus, unlike conventional fine-tuning method where the backbone model is also updated, the Sentence Transformer stays frozen in our case; with only the classification head being trained.

## Emotion Classification (EmoT) with SVC

```sh
python transfer_text_classification.py \
    --model-name LazarusNLP/simcse-indobert-base \
    --dataset-name indonlp/indonlu \
    --dataset-config emot \
    --train-split-name train \
    --test-split-name test \
    --text-column tweet \
    --label-column label
```

### Results

| Model                                                                                                                       | Accuracy (%) | F1 Macro (%) |
| --------------------------------------------------------------------------------------------------------------------------- | :----------: | :----------: |
| [SimCSE-IndoBERT Lite Base](https://huggingface.co/LazarusNLP/simcse-indobert-lite-base)                                    |    41.13     |    40.70     |
| [SimCSE-IndoRoBERTa Base](https://huggingface.co/LazarusNLP/simcse-indoroberta-base)                                        |    50.45     |    50.75     |
| [SimCSE-IndoBERT Base](https://huggingface.co/LazarusNLP/simcse-indobert-base)                                              |    55.45     |    55.78     |
| [ConGen-IndoBERT Lite Base](https://huggingface.co/LazarusNLP/congen-indobert-lite-base)                                    |    58.18     |    58.84     |
| [ConGen-IndoBERT Base](https://huggingface.co/LazarusNLP/congen-indobert-base)                                              |    57.04     |    57.06     |
| [ConGen-SimCSE-IndoBERT Base](https://huggingface.co/LazarusNLP/congen-simcse-indobert-base)                                |    59.54     |    60.37     |
| [SCT-IndoBERT Base](https://huggingface.co/LazarusNLP/sct-indobert-base)                                                    |    61.13     |    61.70     |
| [S-IndoBERT Base mMARCO](https://huggingface.co/LazarusNLP/s-indobert-base-mmarco)                                          |    48.86     |    47.92     |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)   |    63.63     |    64.13     |
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) |    63.18     |    63.78     |
| [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)                                              |    64.54     |    65.04     |
| [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)                                                |    68.63     |    69.07     |
| [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)                                              |    74.77     |    74.66     |

## Sentiment Analysis (SmSA) with SVC

```sh
python transfer_text_classification.py \
    --model-name LazarusNLP/simcse-indobert-base \
    --dataset-name indonlp/indonlu \
    --dataset-config smsa \
    --train-split-name train \
    --test-split-name test \
    --text-column text \
    --label-column label
```

### Results

| Model                                                                                                                       | Accuracy (%) | F1 Macro (%) |
| --------------------------------------------------------------------------------------------------------------------------- | :----------: | :----------: |
| [SimCSE-IndoBERT Lite Base](https://huggingface.co/LazarusNLP/simcse-indobert-lite-base)                                    |     68.8     |    63.37     |
| [SimCSE-IndoRoBERTa Base](https://huggingface.co/LazarusNLP/simcse-indoroberta-base)                                        |     76.2     |    70.42     |
| [SimCSE-IndoBERT Base](https://huggingface.co/LazarusNLP/simcse-indobert-base)                                              |     85.6     |    81.50     |
| [ConGen-IndoBERT Lite Base](https://huggingface.co/LazarusNLP/congen-indobert-lite-base)                                    |     81.2     |    75.59     |
| [ConGen-IndoBERT Base](https://huggingface.co/LazarusNLP/congen-indobert-base)                                              |     85.4     |    82.12     |
| [ConGen-SimCSE-IndoBERT Base](https://huggingface.co/LazarusNLP/congen-simcse-indobert-base)                                |     83.0     |    78.74     |
| [SCT-IndoBERT Base](https://huggingface.co/LazarusNLP/sct-indobert-base)                                                    |     82.0     |    76.92     |
| [S-IndoBERT Base mMARCO](https://huggingface.co/LazarusNLP/s-indobert-base-mmarco)                                          |     80.2     |    75.73     |
| [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)   |     78.8     |    73.64     |
| [paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) |     89.6     |    86.56     |
| [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small)                                              |     83.6     |    79.51     |
| [multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)                                                |     89.4     |    86.22     |
| [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large)                                              |     90.0     |    86.50     |

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
@inproceedings{wilie2020indonlu,
  title={IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding},
  author={Bryan Wilie and Karissa Vincentio and Genta Indra Winata and Samuel Cahyawijaya and X. Li and Zhi Yuan Lim and S. Soleman and R. Mahendra and Pascale Fung and Syafri Bahar and A. Purwarianti},
  booktitle={Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing},
  year={2020}
}
```