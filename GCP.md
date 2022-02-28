# GCP 사용법

참고할 만한 링크
```
gcp 명령어 모음: https://codechacha.com/ko/gcp-command/
GCP 사용법(https://baekyeongmin.github.io/dev/tpu-recipe/)
```
과정 요약 정리
1. 프로젝트 생성 및 설정
2. TPU 노드 생성(메뉴 - compute-engine - TPU - TPU 노드 만들기)
  - 이름, 영역, tpu 유형, tpu 소프트웨어 버전(tensorflow or torch) 설정
  - TFRC 프로그램으로 무료 TPU 이용 시 설정법(**TFRC에서 허가 메일을 받은 후 지역 재확인 필요**):
    - TPU 3: 지역=`europe-west4-a`, 최대 5개 가능
    - TPU 2: 지역=`us-central1-f`, 최대 5개 가능
    - TPU 2 preempitible: `us-central1-f` 최대 100개 가능
3. VM, Storage 설정
  - VM: TPU만 쓰고자 한다면 최소 사양으로 설정. (n1-strandard)
  - Storage: 데이터 및 모델 저장 공간. tpu와 동일한 지역에 만들 것을 권장함.
  - 설정 방법: 메뉴-compute engine - vm 인스턴스

4. tfrecord 파일을 storage에 업로드

5. VM 실행
  5.1. 터미널 창에서 vm을 ssh로 접속
  5.2. 
