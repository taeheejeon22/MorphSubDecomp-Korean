# setting:

# batch: 16, 32
# lr: 5e-5, 3e-5, 2e-5
# epoch: 5

    # # hyperparameters
    # parser.add_argument("--num_epochs", type=int)
    # parser.add_argument("--batch_size", type=int)
    # parser.add_argument("--learning_rate", type=float)


tasks=("korsts" "paws" "cola" "pc")

#best_hparam:
# korsts    32	3e-5	5
# paws	64	3e-5	5
# cola	64	2e-5	5
# pc	32	3e-5	2

# tasks=("korsts" "nsmc" "paws" "cola" "pc" "kornli")

num_epochs=5

echo -e "use_tpu: tpu or gpu ? " 
read use_tpu
echo "use_tpu == $use_tpu"

echo -e "vocab_size: 32k or 64k ? " 
read vocab_size
echo "vocab_size == $vocab_size"

tokenizers=("sp-${vocab_size}" "mecab_orig_composed_sp-${vocab_size}" "mecab_orig_decomposed_pure_sp-${vocab_size}" "mecab_orig_decomposed_morphological_sp-${vocab_size}" 
"mecab_fixed_composed-sp-${vocab_size}" "mecab_fixed_decomposed_pure_sp-${vocab_size}" "mecab_fixed_decomposed_morphological_sp-${vocab_size}")


# 각 배치사이즈, 각 학습률 별로 태스크를 수행함.
# 에포크 수는 5회로 통일.

# 요약본 저장용 directory 생성
# if [ ! -d ./run_outputs/batch_${batch_size}_lr_${learning_rate}/summary_by_hparam ]; then
#     echo "summary_by_hparam dir making..."
#     mkdir -p "./run_outputs/batch_${batch_size}_lr_${learning_rate}/summary_by_hparam"
#     echo "summary_by_hparam dir making...Done"
# fi

# if [ ! -e "./run_outputs/batch_${batch_size}_lr_${learning_rate}/summary_by_hparam/summary_by_hparam.csv" ]; then
#     touch "./run_outputs/batch_${batch_size}_lr_${learning_rate}/summary_by_hparam/summary_by_hparam.csv"
#     chmod +x "./run_outputs/batch_${batch_size}_lr_${learning_rate}/summary_by_hparam/summary_by_hparam.csv"
#     echo "summary_by_hparam file making..."
#     echo "summary_by_hparam file making...Done"
# fi



for task in "${tasks[@]}"; do
    
    # task별 best hparam 할당
    if [[ $task == 'korsts' ]]; then
        batch_size=32
        learning_rate=3e-5
        num_epochs=5

    elif [[ $task == 'paws' ]]; then
        batch_size=64 
        learning_rate=3e-5
        num_epochs=5
    elif [[ $task == 'cola' ]]; then
        batch_size=64 
        learning_rate=2e-5
        num_epochs=5
    elif [[ $task == 'pc' ]]; then
        batch_size=32
        learning_rate=3e-5
        num_epochs=2
    fi
    
    echo "### batch_size: ${batch_size} ###"
    echo "### learning_rate: ${learning_rate} ###"
    echo "### vocab_size: ${vocab_size} ###"
    echo "### task: ${task} ###"
    echo "### use_tpu: ${use_tpu}"

    if [[ $vocab_size == "64k" ]]; then

        for tokenizer in "${tokenizers}"; do
            python3 tasks/$task/run_train.py --tokenizer $tokenizer \
            --resource_dir ./resources/v5_without_dummy_letter \
            --use_tpu $use_tpu \
            --batch_size $batch_size \
            --learning_rate $learning_rate \
            --num_epochs $num_epochs
        done
    else
        echo "vocab_size error!!!"
    fi

done


