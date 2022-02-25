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
ex)
```
./hparam_nsmc.sh
1 # gpu number
```

## 4. log 확인
학습을 진행하는 동안 `run_outputs` 디렉토리에 log가 저장됩니다.
각 seed, batch size, lr 별로 별개의 디렉토리에 저장됩니다.

또한, 모든 세팅의 log를 통합한 파일도 추가로 저장됩니다.
`total_log.csv`: nsmc, paws, hsd, cola의 log가 저장됩니다. 각 칼럼마다 hyperparameter, 토크나이저 정보, dev, test 점수가 저장됩니다.
`klue_total_log.csv`: KLUE task들의 log가 저장됩니다. 
  - 주의: 1 에포크 당 점수가 4번 저장됩니다. 예를 들어 첫 번째 epoch의 값은 4번째 행에 저장되고, 두 번째 epoch의 값은 8번째 행에 저장되는 식입니다. 그렇기 때문에 **epoch 칼럼에 기록되는 숫자는 실제 epoch와 다릅니다.**
  - 만약 klue-dp, klue-sts와 같이 metric을 여러 개로 설정한 경우 각 메트릭마다 1 epoch당 4개의 값이 저장됩니다. 즉, 7번째 행에 첫 번째 메트릭의 1epoch값이 저장되고, 8번째 행에 두 번째 메트릭의 1epoch값이 저장됩니다.

## 5. log 정리(스프레드시트)
5.1. 접속
- https://docs.google.com/spreadsheets/d/1rHuXSwGfvQJciZmJQt2SbnB9kdkFzbMhQ8UAb7jdxMc/edit#gid=1889642085 에 접속합니다.

5.2. 기록 붙여넣기
- `total_log.csv`: 쌓인 로그를 "final_hparam" 탭에 붙여넣기합니다.
- `klue_total_log.csv`: 쌓인 로그를 "final_klue 탭에 붙여넣기합니다.

5.3. 분석
- "final_hparam_table"에서 각 task별 점수가 기록됩니다. 붉은색일수록 점수가 높고, 초록색일수록 점수가 낮습니다. (서식 - 조건부서식 - 색상스케일)
- 각 칸의 점수는 seed 5개 값의 평균을 나타냅니다. (주의: seed5개 결과가 모두 기록되지 않았더라도 평균 값으로 나타남.) 
- dev set 평균 1위인 세팅을 best hyperparameter로 선정합니다. metric이 여러 개인 경우, 각 metric별 평균이 1위인 세팅을 선정합니다.

5.4. 결과 비교
- "final_hparam_table"에 best hyperparameter 값을 입력합니다. (`토크나이저 순서를 "final_hparam_table" 탭과 일치시킨 후 복사 - 붙일 칸에서 마우스 오른쪽 버튼 클릭 - 선택하여 붙여넣기 - 순서 바꾸기` 순으로 붙여넣기를 하면 가로로 나열된 데이터를 세로칸에 붙여넣을 수 있습니다.)
- 각 칼럼별로 점수 순위가 색으로 표현됩니다.
- 굵은 글씨 + 밑줄: 1위, 밑줄: 2위
  - 1위에 굵은글씨+밑줄 적용하기: `서식 - 조건부서식 - 범위 선택 - 형식규칙: '같음', =subtotal(104, 범위) - 서식지정스타일: 굵은글씨, 밑줄`
  - 2위에 밑줄 적용하기: `서식 - 조건부서식 - 범위 선택 - 형식규칙: '맞춤수식', =(rank(첫번째칸, 범위)=2) - 서식지정스타일: 밑줄`


