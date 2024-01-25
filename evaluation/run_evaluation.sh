#!/usr/bin/env bash
model=$1
model_name="${model#*/}"

###############################
# RETRIEVAL
###############################

python retrieval/eval_tydiqa.py \
    --model-name $model \
    --test-dataset-name khalidalt/tydiqa-goldp \
    --test-dataset-config indonesian \
    --test-dataset-split validation \
    --test-batch-size 32 \
    --output-folder retrieval/results/$model_name

python retrieval/eval_miracl.py \
    --model-name $model \
    --test-dataset-name miracl/miracl \
    --test-dataset-config id \
    --test-dataset-split dev \
    --test-batch-size 32 \
    --output-folder retrieval/results/$model_name

###############################
# PAIR CLASSIFICATION
###############################

for split in test_lay test_expert
do
  python pair_classification/eval_pair_classification.py \
      --model-name $model \
      --dataset-name indonli \
      --test-split-name $split \
      --text-column-1 premise \
      --text-column-2 hypothesis \
      --label-column label \
      --output-folder pair_classification/results/$model_name
done

###############################
# CLASSIFICATION
###############################

python classification/eval_classification.py \
    --model-name $model \
    --dataset-name indonlp/indonlu \
    --dataset-config emot \
    --train-split-name train \
    --test-split-name test \
    --text-column tweet \
    --label-column label \
    --output-folder classification/results/$model_name

python classification/eval_classification.py \
    --model-name $model \
    --dataset-name indonlp/indonlu \
    --dataset-config smsa \
    --train-split-name train \
    --test-split-name test \
    --text-column text \
    --label-column label \
    --output-folder classification/results/$model_name

mteb \
    -m $model \
    -l id \
    --output_folder mteb/results/$model_name

###############################
# SEMANTIC TEXTUAL SIMILARITY
###############################

python sts/eval_sts.py \
    --model-name $model \
    --test-dataset-name LazarusNLP/stsb_mt_id \
    --test-dataset-split test \
    --test-text-column-1 text_1 \
    --test-text-column-2 text_2 \
    --test-label-column correlation \
    --test-batch-size 32 \
    --output-folder sts/results/$model_name