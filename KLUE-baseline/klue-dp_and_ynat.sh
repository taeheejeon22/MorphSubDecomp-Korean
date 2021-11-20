#!/bin/bash

set +x

OUTPUT_DIR="../run_outputs"
DATA_DIR="data/klue_benchmark"  # default submodule for data from https://github.com/KLUE-benchmark/KLUE
VERSION="v1.1"

# gpu 입력 받기
echo "gpus 0 1 2 3 ? "
echo -e "gpus: "
read gpus
echo "gpus == ${gpus}"


tasks=("klue-dp" "ynat")

# tokenizers=("eojeol_mecab_fixed_composed_grammatical_symbol_F_wp-64k" 
# "eojeol_mecab_fixed_decomposed_pure_grammatical_symbol_F_wp-64k"
# "morpheme_mecab_orig_composed_grammatical_symbol_F_wp-64k"
# "morpheme_mecab_orig_decomposed_pure_grammatical_symbol_F_wp-64k"
# "morpheme_mecab_fixed_composed_grammatical_symbol_F_wp-64k"
# "morpheme_mecab_fixed_composed_grammatical_symbol_T_wp-64k"
# "morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_T_wp-64k"
# )

tokenizers=("morpheme_mecab_fixed_composed_grammatical_symbol_T_wp-64k"
"morpheme_mecab_fixed_composed_grammatical_symbol_T_wp-64k"
"morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_T_wp-64k"
"morpheme_mecab_fixed_decomposed_pure_grammatical_symbol_T_wp-64k"
"morpheme_mecab_fixed_decomposed_grammatical_grammatical_symbol_T_wp-64k")






# "morpheme_mecab_fixed_decomposed_pure_grammatical_symbol_F_wp-64k"
#
# "morpheme_mecab_fixed_decomposed_pure_grammatical_symbol_T_wp-64k"
#
# "morpheme_mecab_fixed_decomposed_grammatical_grammatical_symbol_T_wp-64k"

# 각 task, resource(tokenizer) 마다 task 수행

for task in "${tasks[@]}"; do
    echo "##### $task #####"

    for tokenizer in "${tokenizers[@]}"; do

        echo "##### ${tokenizer} ##### "
        resources="../resources/v6_without_dummy_letter_grammatical_symbol_T/${tokenizer}"

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
        fi
    done
done


# python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR}
# --data_dir ${DATA_DIR}/${task}-${VERSION}
# --model_name_or_path ${model_name}
# --learning_rate 5e-5 --train_batch_size 32 --warmup_ratio 0.1
# --max_seq_length 128 --patience 100000 --metric_key macro_f1 --gpus 0 --num_workers 4

# python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}
# --model_name_or_path ${model_name}
# --learning_rate 5e-5 --num_train_epochs 10 --warmup_ratio 0.1 --train_batch_size 32 --patience 10000
# --max_seq_length 256 --metric_key las_macro_f1 --gpus 0 --num_workers 4
