# GCP 사용법

참고할 만한 링크
```
gcp 명령어 모음: https://codechacha.com/ko/gcp-command/
GCP 사용법(https://baekyeongmin.github.io/dev/tpu-recipe/)
GCP에서 TPU 사용하기 (https://blog.pingpong.us/tpu-with-tf2-and-gcp/)
```
### 과정 요약 정리
1. 프로젝트 생성 및 설정
2. TPU 노드 생성(메뉴 - compute engine - TPU - TPU 노드 만들기)
    - 이름, 영역, tpu 유형, tpu 소프트웨어 버전(tensorflow or torch) 설정
    - TRC 프로그램으로 무료 TPU 이용 시 설정법(**TRC에서 허가 메일을 받은 후 지역 재확인 필요**):
      - TPU 3: 지역=`europe-west4-a`, 최대 5개 가능
      - TPU 2: 지역=`us-central1-f`, 최대 5개 가능
      - TPU 2 preempitible: `us-central1-f`, 최대 100개 가능
3. VM, Storage 설정
    - VM: TPU만 쓰고자 한다면 최소 사양으로 설정. (n1-strandard)
    - Storage: 데이터 및 모델 저장 공간. tpu와 동일한 지역에 만들 것을 권장함.
    - 설정 방법: 메뉴 - compute engine - vm 인스턴스

4. tfrecord 파일, resource 파일들을 storage에 업로드

5. VM 실행

    5.1. 터미널 창에서 vm에 ssh로 접속
  
    5.2. acl_tokenization에서 `pretrain_code/wp_pretraining_code_stage2.sh` 실행
    
    - (tfrecord 만들기는 `wp_offline_pretraining_code_stage1.sh` 를 로컬 pc에서 실행)
          
          
        - 쉘 스크립트 실행 (첫 실행 시에는 스크립트 첫 번째 단에 있는 `pip install...`을 주석 해제하여 필요한 도구들을 다운로드해야 함.)
        
            `./wp_pretraining_code_stage2.sh` 
    
        - 메시지가 뜨는 대로 tfrecord path, resource path, bert_model을 저장할 path, project name, tpu name, region 입력 

        - 만약 학습 도중에 중단 후, 백업된 checkpoint로부터 학습을 다시 진행할 경우 "init_checkpoints 쓰면 T, 안 쓰면 F" 라는 메시지가 뜰 때 'T'를 입력.
        - T를 입력한 후 checkpoint 숫자 입력

        - 학습 파라미터는 batch=1024, max_seq_length=128, max_predictions_per_seq=20, num_train_steps=1000000, num_warmup_steps=10000, lr=5e-5 로 고정.

        - 만약 checkpoint 저장 간격을 바꾸고자 한타면 스크립트 내에서 `--save_checkpoints_steps` 인자를 수정하면 됨.
        
      5.3. 5.2 과정 이후 학습이 진행됨. 명령어 log는 `command.log`에 저장되고, 학습 로그는 `토크나이저명.log`에 저장됨.


6. 기타 사용법
        
     - gcp 명령어로 스토리지와 로컬, 스토리지와 vm 간 파일 전송
         - 웹상에서 진행해도 되나 `gsutil cp 옮길자료가있는원주소 자료를옮길주소` 를 사용할 수 있음.
         - 이때 gcp 스토리지의 주소를 가리킬 때는 주소 맨 앞에 `gs://` 기입 필요.

     - gcp 명령어로 tpu 만들기
         ```
        (vm)
        gcloud compute tpus create fine-tuning-v3 \
        --zone=europe-west4-a \
        --network=default \
        --version=tensorflow-1.14 \
        --accelerator-type=v3-8
         ```

    - gcp 명령어로 vm 만들기 (예제)
        ```
        (local terminal)
        TPU_NAME="fine-tuning-1"
        ZONE="us-central1-f" 
        PROJECT_ID="smooth-loop-327807"

        gcloud compute --project=${PROJECT_ID} instances create fine-tuning-4 \
        --zone=us-central1-f  \
        --machine-type=n1-standard-1  \
        --boot-disk-size=30GB \
        #--image-family=torch-xla \ # 필수 x
        #--image-project=ml-images  \ # 필수 x
        #--scopes=https://www.googleapis.com/auth/cloud-platform # 필수 x

    - gcp 명령어로 tpu 리스트 확인
        ```
        gcloud compute tpus list --zone=us-central1-f
        ```

7. vm 삭제 및 tpu 삭제
    - 웹에서 진행.
    - 