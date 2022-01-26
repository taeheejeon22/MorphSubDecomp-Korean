#!/bin/bash

#############################
# 하이퍼파라미터를 찾기 위해 각 세팅별로 run_train.py를 반복하는 코드입니다.
# batch_size, learning_rate, epoch 수, task 종류, seed, tokenizer 사용할 gpu를 설정할 수 있습니다.
# 각 하이퍼파라미터에 여러 세팅을 입력하면 입력한 수만큼 반복하여 실행하게 됩니다.
# spacing 옵션을 True로 하여 포함시키게 되면 데이터 전처리 시 띄어쓰기 교정을 수행하게 됩니다. 단, spacing은 nsmc, hsd task에 대해서만 가능합니다.
#############################

# setting:

batch_sizes=(16 32 64)
learning_rates=(1e-5 2e-5 3e-5 5e-5)
num_epochs=5
seeds=(671155)
tasks=("cola" "nsmc" "paws" "hsd")

# 띄어쓰기 교정 적용 여부(nsmc, hsd만 해당). spacing을 사용하려면 아래의 주석 처리를 해제하고, for문 아래의 run_train.py의 주석처리 또한 해제하시면 됩니다.


# 사용할 gpu 선택
echo -e "gpu num 0 1 2 3 ? " 
read gpu_num
echo "gpu_num == ${gpu_num}"

tokenizers=("eojeol_mecab_fixed_composed_grammatical_symbol_F_wp-64k" "eojeol_mecab_fixed_decomposed_pure_grammatical_symbol_F_wp-64k" 
"morpheme_mecab_fixed_composed_grammatical_symbol_F_wp-64k"
"morpheme_mecab_fixed_decomposed_pure_grammatical_symbol_F_wp-64k" "morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64k"
"morpheme_mecab_orig_composed_grammatical_symbol_F_wp-64k" "morpheme_mecab_orig_decomposed_pure_grammatical_symbol_F_wp-64k"
"morpheme_mecab_fixed_decomposed_grammatical_grammatical_symbol_F_wp-64k")


for seed in "${seeds[@]}"; do

    for batch_size in "${batch_sizes[@]}"; do

        for learning_rate in "${learning_rates[@]}"; do

            for task in "${tasks[@]}"; do
                # 띄어쓰기 적용 여부
                if [[ ${task} == "nsmc" ]]; then
                    spacing="spacing"
                else
                    spacing="unspacing"
                fi

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
                    --spacing ${spacing}
                done

            done

        done

    done

done


