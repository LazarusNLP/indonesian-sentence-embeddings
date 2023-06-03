# mMARCO

**[mMARCO](https://huggingface.co/datasets/unicamp-dl/mmarco)** is a multilingual version of the [MS MARCO passage ranking dataset](https://microsoft.github.io/msmarco/), translated via Google Translate API. It supports up to 14 languages, including Indonesian. 

Unlike the original MS MARCO dataset, this version only has query-positive-negative triplets. In the original version, for instance, it had a list of passages which may be relevant to the query, and a label for the most relevant passage.

## Bi-Encoder with MultipleNegativesRankingLoss

### IndoBERT Base

```sh
python train_bi-encoder_mmarco_mnrl.py \
    --model-name indobenchmark/indobert-base-p1 \
    --train-dataset-name unicamp-dl/mmarco \
    --train-dataset-config indonesian \
    --max-seq-length 32 \
    --max-train-samples 1000000 \
    --num-epochs 5 \
    --train-batch-size 128 \
    --learning-rate 2e-5 \
    --warmup-ratio 0.1
```

## Results

### STSB-MT-ID

| Model | Spearman's Correlation (%) | #params | Base Model |
| ----- | :------------------------: | :-----: | ---------- |
|       |                            |         |            |

## References

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