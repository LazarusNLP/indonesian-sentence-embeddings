# ConGen

[ConGen](https://github.com/KornWtp/ConGen) is an unsupervised, knowledge distillation technique that aims to *control* and *generalize* smaller student model from a larger sentence embedding teacher model. In short, the technique enforces the student model to mimic the logits of the teacher model on an instance queue subset of the training data (control) and also generalize it to augmentations of texts for robustness.

Training via ConGen requires an unsupervised corpus, which is readily available for Indonesian texts. In our experiments, we used [Wikipedia texts](https://huggingface.co/datasets/LazarusNLP/wikipedia_id_20230520). As for the data augmentation method, Limkonchotiwat et al. (2022) proposed using back-translation via an NMT model or Google Translate API. However, since that is costly to compute for 1 million texts, we opted for a simple single-word deletion technique.

## ConGen with Single-word Deletion

### IndoBERT Base

```sh
python train_con_gen.py \
    --model-name indobenchmark/indobert-base-p1 \
    --train-dataset-name LazarusNLP/wikipedia_id_20230520 \
    --max-seq-length 32 \
    --max-train-samples 1000000 \
    --num-epochs 20 \
    --train-batch-size 128 \
    --early-stopping-patience 7 \
    --learning-rate 1e-4 \
    --teacher-model-name sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
    --queue-size 65536 \
    --student-temp 0.5 \
    --teacher-temp 0.5
```

### IndoBERT Lite Base

```sh
python train_con_gen.py \
    --model-name indobenchmark/indobert-lite-base-p1 \
    --train-dataset-name LazarusNLP/wikipedia_id_20230520 \
    --max-seq-length 32 \
    --max-train-samples 1000000 \
    --num-epochs 20 \
    --train-batch-size 128 \
    --early-stopping-patience 7 \
    --learning-rate 3e-4 \
    --teacher-model-name sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
    --queue-size 65536 \
    --student-temp 0.05 \
    --teacher-temp 0.05
```

### SimCSE-IndoBERT Base

```sh
python train_con_gen.py \
    --model-name LazarusNLP/simcse-indobert-base \
    --train-dataset-name LazarusNLP/wikipedia_id_20230520 \
    --max-seq-length 32 \
    --max-train-samples 1000000 \
    --num-epochs 20 \
    --train-batch-size 128 \
    --early-stopping-patience 7 \
    --learning-rate 1e-4 \
    --teacher-model-name sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
    --queue-size 65536 \
    --student-temp 0.5 \
    --teacher-temp 0.5
```

### Multilingual e5 Small

```sh
python train_con_gen.py \
    --model-name intfloat/multilingual-e5-small \
    --train-dataset-name LazarusNLP/wikipedia_id_20230520 \
    --max-seq-length 128 --min-text-length 150 --max-text-length 500 \
    --max-train-samples 1000000 \
    --num-epochs 20 \
    --train-batch-size 128 \
    --early-stopping-patience 7 \
    --learning-rate 1e-4 \
    --teacher-model-name sentence-transformers/paraphrase-multilingual-mpnet-base-v2 \
    --queue-size 65536 \
    --student-temp 0.5 \
    --teacher-temp 0.5
```

## ConGen with Cohere Embeddings

### NusaBERT Base

```sh
python train_con_gen_cohere.py \
    --model-name LazarusNLP/NusaBERT-base \
    --train-dataset-name Cohere/wikipedia-2023-11-embed-multilingual-v3 \
    --max-seq-length 128 \
    --max-train-samples 1000000 \
    --num-epochs 20 \
    --train-batch-size 128 \
    --early-stopping-patience 7 \
    --learning-rate 1e-4 \
    --queue-size 65536 \
    --student-temp 0.5 \
    --teacher-temp 0.5
```

## References

```bibtex
@inproceedings{limkonchotiwat-etal-2022-congen,
  title = "{ConGen}: Unsupervised Control and Generalization Distillation For Sentence Representation",
  author = "Limkonchotiwat, Peerat  and
    Ponwitayarat, Wuttikorn  and
    Lowphansirikul, Lalita and
    Udomcharoenchaikit, Can  and
    Chuangsuwanich, Ekapol  and
    Nutanong, Sarana",
  booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
  year = "2022",
  publisher = "Association for Computational Linguistics",
}
```