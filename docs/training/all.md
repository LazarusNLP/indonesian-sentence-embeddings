# All Supervised Datasets

Inspired by [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), we fine-tuned Indonesian sentence embedding models on a set of existing supervised datasets. The tasks included in the training dataset are: question-answering, textual entailment, retrieval, commonsense reasoning, and natural language inference. Currently, our script simply concatenates these datasets and our models are trained conventionally using the `MultipleNegativesRankingLoss`.

## Training Data

| Dataset                                                                                                            |              Task              |                 Data Instance                 | Number of Training Tuples |
| ------------------------------------------------------------------------------------------------------------------ | :----------------------------: | :-------------------------------------------: | :-----------------------: |
| [indonli](https://huggingface.co/datasets/indonli)                                                                 |   Natural Language Inference   |    `(premise, entailment, contradiction)`     |           3,914           |
| [indolem/indo_story_cloze](https://huggingface.co/datasets/indolem/indo_story_cloze)                               |     Commonsense Reasoning      | `(context, correct ending, incorrect ending)` |           1,000           |
| [unicamp-dl/mmarco](https://huggingface.co/datasets/unicamp-dl/mmarco)                                             |       Passage Retrieval        | `(query, positive passage, negative passage)` |          100,000          |
| [miracl/miracl](https://huggingface.co/datasets/miracl/miracl)                                                     |       Passage Retrieval        | `(query, positive passage, negative passage)` |           8,086           |
| [SEACrowd/wrete](https://huggingface.co/datasets/SEACrowd/wrete)                                                   |       Textual Entailment       |           `(sentenceA, sentenceB)`            |            183            |
| [SEACrowd/indolem_ntp](https://huggingface.co/datasets/SEACrowd/indolem_ntp)                                       |       Textual Entailment       |             `(tweet, next tweet)`             |           5,681           |
| [khalidalt/tydiqa-goldp](https://huggingface.co/datasets/khalidalt/tydiqa-goldp)                                   | Extractive Question-Answering  |  `(question, passage)`, `(question, answer)`  |          11,404           |
| [SEACrowd/facqa](https://huggingface.co/datasets/SEACrowd/facqa)                                                   | Extractive Question-Answering  |  `(question, passage)`, `(question, answer)`  |           4,990           |
| *included in v2*                                                                                                   |
| [indonesian-nlp/lfqa_id](https://huggingface.co/datasets/indonesian-nlp/lfqa_id)                                   | Open-domain Question-Answering |             `(question, answer)`              |          226,147          |
| [jakartaresearch/indoqa](https://huggingface.co/datasets/jakartaresearch/indoqa)                                   | Extractive Question-Answering  |  `(question, passage)`, `(question, answer)`  |           6,498           |
| [jakartaresearch/id-paraphrase-detection](https://huggingface.co/datasets/jakartaresearch/id-paraphrase-detection) |           Paraphrase           |       `(sentence, rephrased sentence)`        |           4,076           |
| **Total**                                                                                                          |                                |                                               |        **371,979**        |

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

### Multilingual e5 Small

```sh
python train_all_mnrl.py \
    --model-name intfloat/multilingual-e5-small \
    --max-seq-length 128 \
    --num-epochs 5 \
    --train-batch-size-pairs 384 \
    --train-batch-size-triplets 256 \
    --learning-rate 2e-5
```

## References

```bibtex
@inproceedings{mahendra-etal-2021-indonli,
  title="{I}ndo{NLI}: A Natural Language Inference Dataset for {I}ndonesian",
  author="Mahendra, Rahmad and Aji, Alham Fikri and Louvan, Samuel and Rahman, Fahrurrozi and Vania, Clara",
  booktitle="Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
  month=nov,
  year="2021",
  address="Online and Punta Cana, Dominican Republic",
  publisher="Association for Computational Linguistics",
  url="https://aclanthology.org/2021.emnlp-main.821",
  pages="10511--10527",
}
```

```bibtex
@inproceedings{koto2022cloze,
  title={Cloze evaluation for deeper understanding of commonsense stories in Indonesian},
  author={Koto, Fajri and Baldwin, Timothy and Lau, Jey Han},
  booktitle={Proceedings of the First Workshop on Commonsense Representation and Reasoning (CSRR 2022)},
  pages={8--16},
  year={2022}
}
```

```bibtex
@misc{bonifacio2021mmarco,
  title={mMARCO: A Multilingual Version of MS MARCO Passage Ranking Dataset}, 
  author={Luiz Henrique Bonifacio and Vitor Jeronymo and Hugo Queiroz Abonizio and Israel Campiotti and Marzieh Fadaee and  and Roberto Lotufo and Rodrigo Nogueira},
  year={2021},
  eprint={2108.13897},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```

```bibtex
@article{10.1162/tacl_a_00595,
  author={Zhang, Xinyu and Thakur, Nandan and Ogundepo, Odunayo and Kamalloo, Ehsan and Alfonso-Hermelo, David and Li, Xiaoguang and Liu, Qun and Rezagholizadeh, Mehdi and Lin, Jimmy},
  title="{MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages}",
  journal={Transactions of the Association for Computational Linguistics},
  volume={11},
  pages={1114-1131},
  year={2023},
  month={09},
  issn={2307-387X},
  doi={10.1162/tacl_a_00595},
  url={https://doi.org/10.1162/tacl\_a\_00595},
  eprint={https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00595/2157340/tacl\_a\_00595.pdf},
}
```

```bibtex
@inproceedings{wilie2020indonlu,
  title={IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding},
  author={Wilie, Bryan and Vincentio, Karissa and Winata, Genta Indra and Cahyawijaya, Samuel and Li, Xiaohong and Lim, Zhi Yuan and Soleman, Sidik and Mahendra, Rahmad and Fung, Pascale and Bahar, Syafri and others},
  booktitle={Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing},
  pages={843--857},
  year={2020}
}
```

```bibtex
@article{DBLP:journals/corr/abs-2011-00677,
  author    = {Fajri Koto and
               Afshin Rahimi and
               Jey Han Lau and
               Timothy Baldwin},
  title     = {IndoLEM and IndoBERT: {A} Benchmark Dataset and Pre-trained Language
               Model for Indonesian {NLP}},
  journal   = {CoRR},
  volume    = {abs/2011.00677},
  year      = {2020},
  url       = {https://arxiv.org/abs/2011.00677},
  eprinttype = {arXiv},
  eprint    = {2011.00677},
  timestamp = {Fri, 06 Nov 2020 15:32:47 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2011-00677.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```bibtex
@inproceedings{ruder-etal-2021-xtreme,
  title = "{XTREME}-{R}: Towards More Challenging and Nuanced Multilingual Evaluation",
  author = "Ruder, Sebastian  and
    Constant, Noah  and
    Botha, Jan  and
    Siddhant, Aditya  and
    Firat, Orhan  and
    Fu, Jinlan  and
    Liu, Pengfei  and
    Hu, Junjie  and
    Garrette, Dan  and
    Neubig, Graham  and
    Johnson, Melvin",
  booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
  month = nov,
  year = "2021",
  address = "Online and Punta Cana, Dominican Republic",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2021.emnlp-main.802",
  doi = "10.18653/v1/2021.emnlp-main.802",
  pages = "10215--10245",
}
```

```bibtex
@inproceedings{purwarianti2007machine,
  title={A Machine Learning Approach for Indonesian Question Answering System},
  author={Ayu Purwarianti, Masatoshi Tsuchiya, and Seiichi Nakagawa},
  booktitle={Proceedings of Artificial Intelligence and Applications },
  pages={573--578},
  year={2007}
}
```