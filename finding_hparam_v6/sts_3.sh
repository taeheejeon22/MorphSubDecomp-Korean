#!/bin/bash

#############################
# 하이퍼파라미터를 찾기 위해 각 세팅별로 run_train.py를 반복하는 코드입니다.
# batch_size, learning_rate, epoch 수, task 종류, seed, tokenizer 사용할 gpu를 설정할 수 있습니다.
# 각 하이퍼파라미터에 여러 세팅을 입력하면 입력한 수만큼 반복하여 실행하게 됩니다.

#############################

# setting:

batch_sizes=(32 64)
learning_rates=(1e-5 2e-5 3e-5 5e-5)
num_epochs=5
seeds=(131932)
tasks=("korsts")


# 사용할 gpu 선택
echo -e "gpu num 0 1 2 3 ? " 
read gpu_num
echo "gpu_num == ${gpu_num}"

# 기존
tokenizers=("eojeol_mecab_fixed_composed_grammatical_symbol_F_wp-64k" "eojeol_mecab_fixed_decomposed_pure_grammatical_symbol_F_wp-64k" 
"morpheme_mecab_fixed_composed_grammatical_symbol_F_wp-64k"
"morpheme_mecab_fixed_decomposed_pure_grammatical_symbol_F_wp-64k" "morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64k"
"morpheme_mecab_orig_composed_grammatical_symbol_F_wp-64k" "morpheme_mecab_orig_decomposed_pure_grammatical_symbol_F_wp-64k")

# # LG
# tokenizers=("LG_mecab_fixed_composed_grammatical_symbol_F_wp-64k" "LG_mecab_fixed_decomposed_grammatical_grammatical_symbol_F_wp-64k"
# "LG_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64k" "LG_mecab_fixed_decomposed_pure_grammatical_symbol_F_wp-64k"
# )


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

                    CUDA_VISIBLE_DEVICES=${gpu_num} python ./tasks/$task/run_train.py --tokenizer ${tokenizer} \
                    --resource_dir ${resource} \
                    --batch_size ${batch_size} \
                    --learning_rate ${learning_rate} \
                    --log_dir ${log_dir} \
                    --summary_dir ${summary_dir} \
                    --num_epochs ${num_epochs} \
                    --seed ${seed}
                done

            done

        done

    done

done


