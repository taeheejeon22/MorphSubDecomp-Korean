# Fine tuning 방법

## 1. KLUE-tasks
`run_klue.py` 파일을 다음과 같이 실행합니다.

예시
```
python run_klue.py train \
--task klue-sts \
--output_dir ./output_dir  \
--data_dir KLUE-baseline/data/klue_benchmark/klue-sts-1.1 \
--model_name_or_path model_path \
--tokenizer_name tokenizer_path \
--config_name config_path \
--learning_rate 1e-5 --train_batch_size 16 --num_train_epochs 5 --warmup_ratio 0.1 --patience 100000 \
--max_seq_length 128 --metric_key pearsonr --gpus 0 --num_workers 8 \
--seed ${seed}
```

## 2. nsmc, paws, cola, hsd
`tasks/${실행할 task}/run_train.py` 파일을 다음과 같이 실행합니다.

예시
```
CUDA_VISIBLE_DEVICES=0 python ./tasks/$task/run_train.py --tokenizer ${tokenizer} \
--resource_dir resource_dir \
--batch_size 16 \
--learning_rate 1e-5 \
--log_dir log_dir \
--summary_dir summary_dir \
--num_epochs 5 \
--seed 1234
```

## 3. hyperparameter 찾기
# finding_hparam 디렉토리에 있는 script를 실행하면 각 task별로 여러 개의 hyperparameter 를 설정하고 실험해볼 수 있습니다.