#!/bin/bash

set +x

OUTPUT_DIR="run_outputs"
DATA_DIR="KLUE-baseline/data/klue_benchmark"  # default submodule for data from https://github.com/KLUE-benchmark/KLUE
VERSION="v1.1"


echo "grammatical_symbol T of F? "

echo -e "grammatical_symbol: "
read grammatical_symbol
echo "grammatical_symbol == ${grammatical_symbol}"

echo "아래의 토크나이저 중에서 사용할 토크나이저를 입력하세요. "
echo "eojeol_mecab_fixed_composed_grammatical_symbol_F_wp-64k"
echo "eojeol_mecab_fixed_decomposed_pure_grammatical_symbol_F_wp-64k"
echo "morpheme_mecab_orig_composed_grammatical_symbol_F_wp-64k    morpheme_mecab_orig_decomposed_pure_grammatical_symbol_F_wp-64k"
echo "morpheme_mecab_fixed_composed_grammatical_symbol_F_wp-64k    morpheme_mecab_fixed_decomposed_pure_grammatical_symbol_F_wp-64k"

echo "morpheme_mecab_fixed_composed_grammatical_symbol_T_wp-64k    morpheme_mecab_fixed_decomposed_pure_grammatical_symbol_T_wp-64k"
echo "morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_T_wp-64k    morpheme_mecab_fixed_decomposed_grammatical_grammatical_symbol_T_wp-64k"

echo -e "tokenizer: " 
read tokenizer
echo "tokenizer == ${tokenizer}"


if [[ ${grammatical_symbol} == "F" ]]; then
    resources="resources/v6_without_dummy_letter_grammatical_symbol_F/${tokenizer}"
elif [[ ${grammatical_symbol} == "T" ]]; then
    resources="resources/v6_without_dummy_letter_grammatical_symbol_T/${tokenizer}"
else
    echo "press T or F!!!!"
fi


# task 입력 받기
echo "task ynat or klue-dp? "
echo -e "task: "
read task
echo "task == ${task}"

# gpu 입력 받기
echo "gpus 0 1 2 3 ? "
echo -e "gpus: "
read gpus
echo "gpus == ${gpus}"



if [[ ${task} == "ynat" ]]; then
    python run_klue.py train \
    --task ${task} \
    --output_dir ${OUTPUT_DIR}  \
    --data_dir ${DATA_DIR}/${task}-${VERSION} \
    --model_name_or_path ${resources} \
    --tokenizer_name ${resources} \
    --config_name ${resources} \
    --learning_rate 5e-5 --train_batch_size 32 --warmup_ratio 0.1 --patience 100000 \
    --max_seq_length 128 --metric_key macro_f1 --gpus ${gpus} --num_workers 16

elif [[ ${task} == "klue-dp" ]]; then
    python run_klue.py train \
    --task ${task} \
    --output_dir ${OUTPUT_DIR} \
    --data_dir ${DATA_DIR}/${task}-${VERSION} \
    --model_name_or_path ${resources} \
    --tokenizer_name ${resources} \
    --config_name ${resources} \
    --learning_rate 5e-5 --num_train_epochs 10 --warmup_ratio 0.1 --train_batch_size 32 --patience 10000 \
    --max_seq_length 128 --metric_key las_macro_f1 --gpus ${gpus} --num_workers 16

else
    echo "try again"
fi

# # YNAT
#task="ynat"
# for model_name in "klue/bert-base" "klue/roberta-small" "klue/roberta-base"; do
#     python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION} --model_name_or_path ${model_name} --learning_rate 5e-5 --train_batch_size 32 --warmup_ratio 0.1 --max_seq_length 128 --patience 100000 --metric_key macro_f1 --gpus 0 --num_workers 4
# done

# python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION} --model_name_or_path klue/roberta-large --learning_rate 5e-5 --train_batch_size 32 --warmup_ratio 0.2 --max_seq_length 128 --patience 100000 --metric_key macro_f1 --gpus 0 --num_workers 4


# task="klue-dp"
# for model_name in "klue/roberta-small" "klue/roberta-base" "klue/bert-base"; do
#     python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path ${model_name} --learning_rate 5e-5 --num_train_epochs 10 --warmup_ratio 0.1 --train_batch_size 32 --patience 10000 --max_seq_length 256 --metric_key las_macro_f1 --gpus 0 --num_workers 4
# done

# python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path klue/roberta-large --learning_rate 5e-5 --num_train_epochs 15 --warmup_ratio 0.2 --train_batch_size 32 --patience 10000 --max_seq_length 256 --metric_key uas_macro_f1 --gpus 0 --num_workers 4