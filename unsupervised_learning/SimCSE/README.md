# SimCSE

Unsupervised [SimCSE](https://github.com/princeton-nlp/SimCSE) is a contrastive learning framework that proposes the usage of different dropout masks as means to generate augmented representations of the same text. There is also a supervised variant of SimCSE that leverages annoated pairs from NLI datasets, using the same contrastive learning framework.

Training via SimCSE requires an unsupervised corpus, which is readily available for Indonesian texts. In our experiments, we used [Wikipedia texts](https://huggingface.co/datasets/LazarusNLP/wikipedia_id_20230520). We used the [Sentence Transformer implementation](https://www.sbert.net/examples/unsupervised_learning/README.html#simcse) of SimCSE.

## Unsupervised SimCSE with MultipleNegativesRankingLoss

### IndoBERT Base

```sh
python train_sim_cse.py \
    --model-name indobenchmark/indobert-base-p1 \
    --train-dataset-name LazarusNLP/wikipedia_id_20230520 \
    --max-train-samples 1000000 \
    --max-seq-length 32 \
    --num-epochs 1 \
    --train-batch-size 128 \
    --learning-rate 3e-5
```

### IndoBERT Lite Base

```sh
python train_sim_cse.py \
    --model-name indobenchmark/indobert-lite-base-p1 \
    --train-dataset-name LazarusNLP/wikipedia_id_20230520 \
    --max-train-samples 1000000 \
    --max-seq-length 75 \
    --num-epochs 1 \
    --train-batch-size 128 \
    --learning-rate 3e-5
```

### IndoRoBERTa Base

```sh
python train_sim_cse.py \
    --model-name flax-community/indonesian-roberta-base \
    --train-dataset-name LazarusNLP/wikipedia_id_20230520 \
    --max-train-samples 1000000 \
    --max-seq-length 32 \
    --num-epochs 1 \
    --train-batch-size 128 \
    --learning-rate 3e-5
```

## Results

### STSB-MT-ID

| Model                                                                                    | Spearman's Correlation (%) | #params | Base Model                                                                        | Train Dataset                                                                 |
| ---------------------------------------------------------------------------------------- | :------------------------: | :-----: | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [SimCSE-IndoBERT Lite Base](https://huggingface.co/LazarusNLP/simcse-indobert-lite-base) |           44.08            |   12M   | [IndoBERT Lite Base](https://huggingface.co/indobenchmark/indobert-lite-base-p1)  | [Wikipedia](https://huggingface.co/datasets/LazarusNLP/wikipedia_id_20230520) |
| [SimCSE-IndoRoBERTa Base](https://huggingface.co/LazarusNLP/simcse-indoroberta-base)     |           61.26            |  125M   | [IndoRoBERTa Base](https://huggingface.co/flax-community/indonesian-roberta-base) | [Wikipedia](https://huggingface.co/datasets/LazarusNLP/wikipedia_id_20230520) |
| [SimCSE-IndoBERT Base](https://huggingface.co/LazarusNLP/simcse-indobert-base)           |           70.13            |  125M   | [IndoBERT Base](https://huggingface.co/indobenchmark/indobert-base-p1)            | [Wikipedia](https://huggingface.co/datasets/LazarusNLP/wikipedia_id_20230520) |

## References

```bibtex
@inproceedings{gao2021simcse,
  title={{SimCSE}: Simple Contrastive Learning of Sentence Embeddings},
  author={Gao, Tianyu and Yao, Xingcheng and Chen, Danqi},
  booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
  year={2021}
}
```