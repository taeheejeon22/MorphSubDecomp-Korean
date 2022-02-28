# Fine tuning 방법 (하나씩 실행할 경우)
- KLUE tasks 와 "nsmc, paws, cola, hsd" tasks의 실행 방법이 각각 다릅니다.
 
## 1. KLUE-tasks
`run_klue.py` 파일을 다음과 같이 실행합니다. 
각 인자에 원하는 하이퍼파라미터를 입력합니다.

예시
```
python run_klue.py train \
--task klue-sts \
--output_dir ./output_dir  \
--data_dir ./KLUE-baseline/data/klue_benchmark/klue-sts-1.1 \
--model_name_or_path ./model_path \
--tokenizer_name ./tokenizer_path \
--config_name ./config_path \
--learning_rate 1e-5 --train_batch_size 16 --num_train_epochs 5 --warmup_ratio 0.1 --patience 100000 \
--max_seq_length 128 --metric_key pearsonr --gpus 0 --num_workers 8 \
--seed ${seed}
```

## 2. nsmc, paws, cola, hsd
`tasks/${실행할 task}/run_train.py` 파일을 다음과 같이 실행합니다. 각 인자에 원하는 하이퍼파라미터를 입력합니다.

예시
```
CUDA_VISIBLE_DEVICES=0 python ./tasks/$task/run_train.py \
--tokenizer ${tokenizer} \
--resource_dir ./resource_dir \
--batch_size 16 \
--learning_rate 1e-5 \
--log_dir ./log_dir \   # dev, test 로그 저장 
--summary_dir ./summary_dir \   # tensorboard용 summary 파일 저장
--num_epochs 5 \
--seed 1234
```

## 3. hyperparameter 찾기 
finding_hparam 디렉토리에 있는 shell script를 실행하면 각 task별로 여러 개의 hyperparameter를 설정하고 실험해볼 수 있습니다.
- (seed는 `random_seeds` 파일에 있습니다.)


ex)
```
./finding_hparam/hparam_nsmc.sh
1 
# 사용할 gpu number 입력
```


## 4. log 확인
학습을 진행하는 동안 `run_outputs` 디렉토리에 log가 저장됩니다.
각 seed, batch size, lr 별로 별개의 디렉토리에 저장됩니다.

또한, 모든 세팅의 log를 통합하여 기록하는 파일도 추가로 저장됩니다.
- `total_log.csv`: nsmc, paws, hsd, cola의 log가 저장됩니다. 각 칼럼마다 hyperparameter, 토크나이저 정보, dev, test 점수가 저장됩니다.
- `klue_total_log.csv`: KLUE task들의 log가 저장됩니다. 
  - **주의**: **KLUE의 경우 1 에포크 당 점수가 4번 저장됩니다.** 예를 들어 첫 번째 epoch의 값은 4번째 행에 저장되고, 두 번째 epoch의 값은 8번째 행에 저장되는 식입니다. 그렇기 때문에 **epoch 칼럼에 기록되는 숫자는 실제 epoch와 다릅니다.**
  - 만약 klue-dp, klue-sts와 같이 metric을 여러 개로 설정한 경우 각 메트릭마다 1 epoch당 4개의 값이 저장됩니다. 즉, 7번째 행에 첫 번째 메트릭의 1epoch값이 저장되고, 8번째 행에 두 번째 메트릭의 1epoch값이 저장됩니다.


## 5. log 정리(스프레드시트)
  5.1. 접속
  - https://docs.google.com/spreadsheets/d/1rHuXSwGfvQJciZmJQt2SbnB9kdkFzbMhQ8UAb7jdxMc/edit#gid=1889642085 에 접속합니다.

  5.2. 기록 붙여넣기
  - (현재 사용하고 있는 탭은 모두 탭 이름 아래 빨간 밑줄이 그어져 있습니다.)
  - `total_log.csv`: 쌓인 로그를 "final_hparam" 탭에 붙여넣기합니다.
  - `klue_total_log.csv`: 쌓인 로그를 "final_klue" 탭에 붙여넣기합니다.
  - (이전 실험의 기록들이 있는 탭은 전부 '숨기기'처리가 되어있습니다. 숨겨진 탭을 보려면 창 하단의 가로줄 4개짜리 버튼을 누른 후 원하는 탭을 클릭하면 됩니다.)


  5.3. 분석
  - "final_hparam_table"에서 각 task별 점수가 기록됩니다. 붉은색일수록 점수가 높고, 초록색일수록 점수가 낮습니다. (서식 - 조건부서식 - 색상스케일)
  - 각 칸의 점수는 seed 5개 값의 평균을 나타냅니다. (주의: seed5개 결과가 모두 기록되지 않았더라도, 현재까지 진행된 seed들의 평균 값으로 나타남.) 
  - **dev set 평균 1위인 세팅**을 best hyperparameter로 선정합니다. **metric이 여러 개인 경우, 각 metric별 평균이 1위인 세팅**을 선정합니다.

  5.4. 결과 비교
  - "final_hparam_table"에 best hyperparameter 값을 입력합니다. (표의 토크나이저 순서를 "final_hparam_table" 탭과 일치시킨 후, 해당 파라미터 세팅의 점수들이 있는 행을 `복사 - 붙일 칸에서 마우스 오른쪽 버튼 클릭 - 선택하여 붙여넣기 - 순서 바꾸기` 순으로 붙여넣기를 하면 가로로 나열된 데이터를 세로칸에 붙여넣을 수 있습니다.)
  - 각 칼럼별로 점수 순위가 색으로 표현됩니다.
  - 굵은 글씨 + 밑줄: 1위, 밑줄: 2위
    - 1위에 굵은글씨+밑줄 적용하기: `서식 - 조건부서식 - 범위 선택 - 형식규칙: '같음', =subtotal(104, 범위) - 서식지정스타일: 굵은글씨, 밑줄`
    - 2위에 밑줄 적용하기: `서식 - 조건부서식 - 범위 선택 - 형식규칙: '맞춤수식', =(rank(첫번째칸, 범위)=2) - 서식지정스타일: 밑줄`


## 기타 수정한 사항
- 원활한 finetuning을 위해 원본 코드를 수정한 내역
  
  1. `KLUE-baseline/klue_baseline/utils/logging.py` line 67
    - `klue_total_log.csv` 기록을 위해 코드를 추가함.
      ```
      # # for total_log
      if k in total_log_keys:
          print('##### k: ', k)
          if os.path.isfile('./run_outputs/klue_total_log.csv') == False:
              with open ('./run_outputs/klue_total_log.csv', 'w', newline="") as f:
                  wr = csv.writer(f)
                  ...
                  ...
      ```

  2. `tasks/cola/trainer.py` 및 hsd, paws 등의 tasks의 train.py 파일들
    - `total_log.csv` 기록을 위한 코드 추가
      ```
      # dev,test 결과만 따로 저장
      self.begin_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
      tokenizer_dir = os.path.join(self.config.resource_dir, self.config.tokenizer)
      self.pretrained_bert_files = [file for file in os.listdir(tokenizer_dir) if file.endswith("pth")]
      self.pretrained_bert_file_name = self.pretrained_bert_files[0]

      if os.path.isfile('./run_outputs/total_log.csv') == False:
          with open ('./run_outputs/total_log.csv', 'w', newline="") as f:
              wr = csv.writer(f)
              ...
              ...
      ```

  3. `tasks/cola/trainer.py` 및 hsd, paws 등의 tasks의 train.py 파일들
    - (과거) TPU 사용 fine tuning을 위한 코드
    - TPU로 fine tuning 학습 시 필요한 부분 (cola, paws 등에만 추가됨. klue는 오리지널 코드에 이미 tpu 사용 코드가 있음.)
    - `run_train.py` 실행 시 --use_tpu 인자에 `tpu`를 넣어주게 되면 실행됨. (단, 현재 코드로는 1개의 코어로 학습이 수행됨.)
      ```
      if self.config.use_tpu == "tpu":
        # optimizer for TPU (Note: Cloud TPU-specific code!)
        import torch_xla.core.xla_model as xm # for using tpu
        xm.optimizer_step(self.optimizer, barrier=True) # multi core 사용 시 barrier=True 불필요
      else:
          self.optimizer.step()

      #self.optimizer.step()
      self.scheduler.step()
      ```

  4. nsmc spacing
    - `tasks/nsmc/data_utils.py` line 5-7, line 26-31, `tasks/nsmc/config.py` line 11
    - 주석 처리를 해제하고, `run_train.py`를 돌릴 때 `--spacing` 인자에 "spacing"을 넣으면 데이터셋에 대해서 띄어쓰기 교정을 먼저 수행한 후 토크나이징 및 학습을 시행함.
    - 또는, `tasks/nsmc/config.py` line 51, 53, 55에서 `train_path`, `dev_path`, `test_path` 를 `ratings...tsv` -> `spaced_ratings..tsv`로 수정한 후 학습 진행. 이 방법이 위의 방법보다 훨씬 속도가 빠름.