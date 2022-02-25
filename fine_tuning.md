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
finding_hparam 디렉토리에 있는 script를 실행하면 각 task별로 여러 개의 hyperparameter를 설정하고 실험해볼 수 있습니다.

## 4. log 확인
학습을 진행하는 동안 `run_outputs` 디렉토리에 log가 저장됩니다.
각 seed, batch size, lr 별로 별개의 디렉토리에 저장됩니다.

또한, 모든 세팅의 log를 통합한 파일도 추가로 저장됩니다.
`total_log.csv`: nsmc, paws, hsd, cola의 log가 저장됩니다. 각 칼럼마다 hyperparameter, 토크나이저 정보, dev, test 점수가 저장됩니다.
`klue_total_log.csv`: KLUE task들의 log가 저장됩니다. 
  - 주의: 1 에포크 당 점수가 4번 저장됩니다. 예를 들어 첫 번째 epoch의 값은 4번째 행에 저장되고, 두 번째 epoch의 값은 8번째 행에 저장되는 식입니다. 그렇기 때문에 **epoch 칼럼에 기록되는 숫자는 실제 epoch와 다릅니다.**
  - 만약 klue-dp, klue-sts와 같이 metric을 여러 개로 설정한 경우 각 메트릭마다 1 epoch당 4개의 값이 저장됩니다. 즉, 7번째 행에 첫 번째 메트릭의 1epoch값이 저장되고, 8번째 행에 두 번째 메트릭의 1epoch값이 저장됩니다.
