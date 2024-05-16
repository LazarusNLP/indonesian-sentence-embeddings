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

##############################
# SEMANTIC TEXTUAL SIMILARITY
##############################

python sts/eval_sts.py \
    --model-name $model \
    --test-dataset-name LazarusNLP/stsb_mt_id \
    --test-dataset-split test \
    --test-text-column-1 text_1 \
    --test-text-column-2 text_2 \
    --test-label-column correlation \
    --test-batch-size 32 \
    --output-folder sts/results/$model_name

###############################
# MTEB TASKS
###############################

for lang in id ind ind-Latn
do
    mteb \
        -m $model \
        -l $lang \
        --output_folder mteb/results/$model_name/ind
done

for lang in jv jav jav-Latn
do
    mteb \
        -m $model \
        -l $lang \
        --output_folder mteb/results/$model_name/jav
done

for lang in sun sun-Latn
do
    mteb \
        -m $model \
        -l $lang \
        --output_folder mteb/results/$model_name/sun
done

for lang in ace ace-Latn
do
    mteb \
        -m $model \
        -l $lang \
        --output_folder mteb/results/$model_name/ace
done

for lang in ban ban-Latn
do
    mteb \
        -m $model \
        -l $lang \
        --output_folder mteb/results/$model_name/ban
done

for lang in bbc
do
    mteb \
        -m $model \
        -l $lang \
        --output_folder mteb/results/$model_name/bbc
done

for lang in bjn bjn-Latn
do
    mteb \
        -m $model \
        -l $lang \
        --output_folder mteb/results/$model_name/bjn
done

for lang in bug bug-Latn
do
    mteb \
        -m $model \
        -l $lang \
        --output_folder mteb/results/$model_name/bug
done

for lang in mad
do
    mteb \
        -m $model \
        -l $lang \
        --output_folder mteb/results/$model_name/mad
done

for lang in min min-Latn
do
    mteb \
        -m $model \
        -l $lang \
        --output_folder mteb/results/$model_name/min
done

for lang in nij
do
    mteb \
        -m $model \
        -l $lang \
        --output_folder mteb/results/$model_name/nij
done