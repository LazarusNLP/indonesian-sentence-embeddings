# SCT

## SCT Distillation with Back-translated Corpus

### 

```sh
python train_sct_distillation.py \
    --model-name indobenchmark/indobert-base-p1 \
    --train-dataset-name LazarusNLP/wikipedia_id_backtranslated \
    --max-seq-length 128 \
    --num-epochs 20 \
    --train-batch-size 128 \
    --early-stopping-patience 7 \
    --learning-rate 1e-4 \
    --teacher-model-name sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
    --queue-size 65536 \
    --student-temp 0.5 \
    --teacher-temp 0.5
```

## Results

## References

```bibtex
@article{10.1162/tacl_a_00620,
    author = {Limkonchotiwat, Peerat and Ponwitayarat, Wuttikorn and Lowphansirikul, Lalita and Udomcharoenchaikit, Can and Chuangsuwanich, Ekapol and Nutanong, Sarana},
    title = "{An Efficient Self-Supervised Cross-View Training For Sentence Embedding}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {11},
    pages = {1572-1587},
    year = {2023},
    month = {12},
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00620},
    url = {https://doi.org/10.1162/tacl\_a\_00620},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00620/2196817/tacl\_a\_00620.pdf},
}
```