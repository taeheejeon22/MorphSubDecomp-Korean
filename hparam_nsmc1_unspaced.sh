#!/bin/bash

# setting:
batch_sizes=(64)
learning_rates=(2e-5)
num_epochs=2
tasks=("nsmc")
# seeds=(670488 116740 26226 777573 288390)
seeds=(670488 116740)
#spacing=False
# tasks=("korsts" "nsmc" "paws" "cola" "pc" "kornli")

# 사용할 gpu 선택
echo -e "gpu num 0 1 2 3 ? " 
read gpu_num
echo "gpu_num == ${gpu_num}"

tokenizers=("morpheme_mecab_fixed_decomposed_pure_grammatical_symbol_T_wp-64k" "morpheme_mecab_fixed_composed_grammatical_symbol_F_wp-64k"
"morpheme_mecab_fixed_decomposed_pure_grammatical_symbol_F_wp-64k" "morpheme_mecab_fixed_composed_grammatical_symbol_T_wp-64k"
"morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_T_wp-64k" "morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64k"
"morpheme_mecab_orig_composed_grammatical_symbol_F_wp-64k" "morpheme_mecab_orig_decomposed_pure_grammatical_symbol_F_wp-64k"
"morpheme_mecab_fixed_decomposed_grammatical_grammatical_symbol_T_wp-64k" "morpheme_mecab_fixed_decomposed_grammatical_grammatical_symbol_F_wp-64k")


for seed in "${seeds[@]}"; do

    for batch_size in "${batch_sizes[@]}"; do

        for learning_rate in "${learning_rates[@]}"; do

            for task in "${tasks[@]}"; do
                log_dir="./run_outputs/batch_"${batch_size}"_lr_"${learning_rate}/$task/logs
                summary_dir="./run_outputs/batch_"${batch_size}"_lr_"${learning_rate}/$task/summaries
                
                echo "### batch_size: ${batch_size} ###"
                echo "### learning_rate: ${learning_rate} ###"
                echo "### vocab_size: ${vocab_size} ###"
                echo "### task: ${task} ###"
                echo "### log_dir: ${log_dir} ###"
                echo "### summary_dir: ${summary_dir} ###"
                echo "### seed: ${seed} ###"
                
                for tokenizer in "${tokenizers[@]}"; do
                    echo "### tokenizer: ${tokenizer} ###"

                    # resource dir
                    if [[ `echo "${tokenizer: (-8):1}"` == "T" ]]; then
                        resource="./resources/v6_without_dummy_letter_grammatical_symbol_T"
                    elif [[ `echo "${tokenizer: (-8):1}"` == "F" ]]; then
                        resource="./resources/v6_without_dummy_letter_grammatical_symbol_F"
                    else
                        echo "tokenizer_name ERROR"
                    fi

                    CUDA_VISIBLE_DEVICES=${gpu_num} python3 tasks/$task/run_train.py --tokenizer ${tokenizer} \
                    --resource_dir ${resource} \
                    --batch_size ${batch_size} \
                    --learning_rate ${learning_rate} \
                    --log_dir ${log_dir} \
                    --summary_dir ${summary_dir} \
                    --num_epochs ${num_epochs} \
                    --seed ${seed} \
#                    --spacing $spacing
                done

            done

        done

    done

done



