#!/bin/bash

# setting:
batch_sizes=(16 32 64)
learning_rates=(1e-5 2e-5 3e-5 5e-5)
tasks=("klue-dp")
seeds=(121958 671155 131932 365838 259178)

num_epochs=10

# 사용할 gpu 선택
echo -e "gpu num 0 1 2 3 ? " 
read gpu_num
echo "gpu_num == ${gpu_num}"

tokenizers=("LG_mecab_fixed_decomposed_grammatical_grammatical_symbol_F_wp-32k" "morpheme_mecab_fixed_decomposed_grammatical_grammatical_symbol_F_wp-32k")

# klue 경로
OUTPUT_DIR="./run_outputs"
DATA_DIR="./KLUE-baseline/data/klue_benchmark"  # default submodule for data from https://github.com/KLUE-benchmark/KLUE
VERSION="v1.1"

# 각 배치사이즈, 각 학습률 별로 태스크를 수행함.
# 에포크 수는 5회로 통일.

for seed in "${seeds[@]}"; do

    for batch_size in "${batch_sizes[@]}"; do

        for learning_rate in "${learning_rates[@]}"; do

            for task in "${tasks[@]}"; do

                echo "### batch_size: ${batch_size} ###"
                echo "### learning_rate: ${learning_rate} ###"
                echo "### vocab_size: ${vocab_size} ###"
                echo "### task: ${task} ###"
                echo "### log_dir: $log_dir ###"
                echo "### summary_dir: $summary_dir ###"
                echo "### seed: ${seed} ###"
                
                for tokenizer in "${tokenizers[@]}"; do
                    echo "### tokenizer: ${tokenizer} ###"

                    # resource dir
                    if [[ `echo "${tokenizer: (-8):1}"` == "T" ]]; then
                        resource="./resources/v7_without_dummy_letter_grammatical_symbol_T"
                    elif [[ `echo "${tokenizer: (-8):1}"` == "F" ]]; then
                        resource="./resources/v7_without_dummy_letter_grammatical_symbol_F"
                    else
                        echo "tokenizer_name ERROR"
                    fi

                    python ./run_klue.py train \
                    --task ${task} \
                    --output_dir ${OUTPUT_DIR}  \
                    --data_dir ${DATA_DIR}/${task}-${VERSION} \
                    --model_name_or_path ${resource}/${tokenizer} \
                    --tokenizer_name ${resource}/${tokenizer} \
                    --config_name ${resource}/${tokenizer} \
                    --learning_rate ${learning_rate} --train_batch_size ${batch_size} --num_train_epochs ${num_epochs} --warmup_ratio 0.1 --patience 100000 \
                    --max_seq_length 128 --metric_key uas_macro_f1 --gpus ${gpu_num} --num_workers 32 \
                    --seed ${seed}

                done

            done

        done

    done

done

