# Evaluation

## Machine Translated STS-B

To the best of our knowledge, there is no official benchmark on Indonesian sentence embeddings. Inspired by [Thai Sentence Vector Benchmark](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark), we translated the [STS-B](https://github.com/facebookresearch/SentEval) dev and test set to Indonesian via Google Translate API. This dataset will be used to evaluate our model's Spearman correlation score on the translated test set. You can find the translated dataset on [🤗 HuggingFace Hub](https://huggingface.co/datasets/LazarusNLP/stsb_mt_id).

For practical purposes, we used Sentence Transformer's [`EmbeddingSimilarityEvaluator`](https://www.sbert.net/docs/package_reference/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator) to perform inference and evaluate our models.

### Example

```sh
python sts/eval_sts.py \
    --model-name LazarusNLP/congen-indobert-base \
    --test-dataset-name LazarusNLP/stsb_mt_id \
    --test-dataset-split test \
    --test-text-column-1 text_1 \
    --test-text-column-2 text_2 \
    --test-label-column correlation \
    --test-batch-size 32
```

## MIRACL (Multilingual Information Retrieval Across a Continuum of Languages)

MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) is a multilingual retrieval dataset that focuses on search across 18 different languages, which collectively encompass over three billion native speakers around the world. We evaluated our models on the Indonesian subset of MIRACL.

We used Sentence Transformer's [`InformationRetrievalEvaluator`](https://www.sbert.net/docs/package_reference/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator) to perform inference and evaluate our models.

### Example

```sh
python retrieval/eval_miracl.py \
    --model-name LazarusNLP/congen-simcse-indobert-base \
    --test-dataset-name miracl/miracl \
    --test-dataset-config id \
    --test-dataset-split dev \
    --test-batch-size 32 \
    --output-folder retrieval/results/congen-simcse-indobert-base
```

## TyDiQA

TyDi QA is a question answering dataset covering 11 typologically diverse languages with 204K question-answer pairs. We evaluated our models on the Indonesian subset of MIRACL.

We used Sentence Transformer's [`InformationRetrievalEvaluator`](https://www.sbert.net/docs/package_reference/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator) to perform inference and evaluate our models.

### Example

```sh
python retrieval/eval_tydiqa.py \
    --model-name LazarusNLP/congen-simcse-indobert-base \
    --test-dataset-name khalidalt/tydiqa-goldp \
    --test-dataset-config indonesian \
    --test-dataset-split validation \
    --test-batch-size 32 \
    --output-folder retrieval/results/congen-simcse-indobert-base
```

## Massive Text Embedding Benchmark (MTEB)

The Massive Text Embedding Benchmark (MTEB) aims to provide clarity on how models perform on a variety of embedding tasks and thus serves as the gateway to finding universal text embeddings applicable to a variety of tasks. We evaluated our models on Indonesian subsets of MTEB that consists of two classification subsets: `MassiveIntentClassification (id)` and `MassiveScenarioClassification (id)`.

We used the official [MTEB](https://github.com/embeddings-benchmark/mteb.git) code to automatically evaluate our models and ensure fairness across evaluations. Follow the official installation instructions found in the MTEB repository.

### Example

```sh
mteb \
    -m LazarusNLP/congen-simcse-indobert-base \
    -l id  \
    --output_folder mteb/results/congen-simcse-indobert-base
```

## IndoNLU Text Classification

Normally, sentence embedding models are leveraged for downstream tasks such as information retrieval, semantic search, clustering, etc. Text classification could similarly leverage these models' sentence embedding capabilities.

We will be doing emotion classification and sentiment analysis on the EmoT and SmSA subsets of [IndoNLU](https://huggingface.co/datasets/indonlp/indonlu), respectively. To do so, we will be doing the same approach as [Thai Sentence Vector Benchmark](https://github.com/mrpeerat/Thai-Sentence-Vector-Benchmark) and simply fit a Linear SVC on sentence representations of our texts with their corresponding labels. Thus, unlike conventional fine-tuning method where the backbone model is also updated, the Sentence Transformer stays frozen in our case; with only the classification head being trained.

### Example: Emotion Classification

```sh
python classification/eval_classification.py \
    --model-name LazarusNLP/simcse-indobert-base \
    --dataset-name indonlp/indonlu \
    --dataset-config emot \
    --train-split-name train \
    --test-split-name test \
    --text-column tweet \
    --label-column label
```

### Example: Sentiment Analysis

```sh
python classification/eval_classification.py \
    --model-name LazarusNLP/simcse-indobert-base \
    --dataset-name indonlp/indonlu \
    --dataset-config smsa \
    --train-split-name train \
    --test-split-name test \
    --text-column text \
    --label-column label
```

## IndoNLI Pair Classification

We can similarly leverage sentence embedding models for zero-shot pair classification tasks. In our case, we will be doing zero-shot natural language inference on the [IndoNLI](https://huggingface.co/datasets/indonli) dataset. We follow the same evaluation procedure as done on the MTEB PairClassification Benchmark. We drop all neutral pairs and re-mapped contradictions as `0`s and entailments as `1`s. Afterwards, we will search for the best threshold values and find the maximum average precision (AP) score, which also serves as the evaluation metric.

### Example

```sh
for split in test_lay test_expert
do
  python pair_classification/eval_pair_classification.py \
      --model-name LazarusNLP/simcse-indobert-base \
      --dataset-name indonli \
      --test-split-name $split \
      --text-column-1 premise \
      --text-column-2 hypothesis \
      --label-column label
done
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
@article{10.1162/tacl_a_00595,
  author = {Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
  title = "{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}",
  journal = {Transactions of the Association for Computational Linguistics},
  volume = {11},
  pages = {1114-1131},
  year = {2023},
  month = {09},
  issn = {2307-387X},
  doi = {10.1162/tacl_a_00595},
  url = {https://doi.org/10.1162/tacl\_a\_00595},
  eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00595/2157340/tacl\_a\_00595.pdf},
}
```

```bibtex
@article{muennighoff2022mteb,
  doi = {10.48550/ARXIV.2210.07316},
  url = {https://arxiv.org/abs/2210.07316},
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Lo{\"\i}c and Reimers, Nils},
  title = {MTEB: Massive Text Embedding Benchmark},
  publisher = {arXiv},
  journal={arXiv preprint arXiv:2210.07316},  
  year = {2022}
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

```bibtex
@inproceedings{mahendra-etal-2021-indonli,
    itle = "{I}ndo{NLI}: A Natural Language Inference Dataset for {I}ndonesian",
  author = "Mahendra, Rahmad and Aji, Alham Fikri and Louvan, Samuel and Rahman, Fahrurrozi and Vania, Clara",
  booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
  month = nov,
  year = "2021",
  address = "Online and Punta Cana, Dominican Republic",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2021.emnlp-main.821",
  pages = "10511--10527",
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