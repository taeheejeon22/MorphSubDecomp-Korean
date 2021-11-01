#!/bin/bash

# 코퍼스, vocab.txt를 바탕으로 tfrecord 파일을 만드는 코드
# 각 토크나이저로 토큰화한 코퍼스 파일들이 있는 폴더에 있는 모든 파일들을 차례로 불러온 뒤,
# 각 파일에 대한 tfrecord 파일을 만든다.


set -e

# 필요한 도구 받기
#pip install tensorflow==1.14
#pip install -U gast==0.2.2
#git clone https://github.com/google-research/bert.git
#bert-sentencepiece version
#pip install sentencepiece==0.1.96
#git clone https://github.com/raymondhs/bert-sentencepiece.git


# tokenizer를 사용자로부터 입력 받기
echo "아래의 토크나이저 중에서 사용할 토크나이저를 입력하세요. "
echo "none_composed"
echo "orig_composed    orig_decomposed_pure    orig_deocomposed_morphological"
echo "fixed_composed    fixed_decomposed_pure    fixed_decomposed_morphological"

echo -e "tokenizer: " 
read TOKENIZER
echo "tokenizer == $TOKENIZER"

echo "corpus_dir을 입력하세요."
echo -e "corpus_dir: "
read CORPUS_DIR
echo "corpus_dir == $CORPUS_DIR"

echo "resource_dir을 입력하세요. "
echo -e "resource_dir: "
read RESOURCE_DIR
echo "resource_dir == $RESOURCE_DIR"


# 입력 받은 tokenizer, corpus의 output_dir

OUTPUT_DIR=`echo ${CORPUS_DIR//"tokenized_GCP"/"tfrecord"}`


# 각 코퍼스 파일에 대해서 tfrecord 만들기

# corpus_files=`gsutil ls gs://$CORPUS_DIR`
# file_count=`gsutil ls gs://$CORPUS_DIR | wc -l`

# echo corpus files: $corpus_files
# echo 코퍼스 파일 수: $file_count

# tok.model을 vm으로 불러오기
gsutil cp gs://$RESOURCE_DIR/tok.model $TOKENIZER'_'tok.model

file_num=0

for file in `gsutil ls gs://$CORPUS_DIR/*`
do
    # if [ "${file}" == "*_[0-9][0-9]" ] || [ "${file}" == "*.txt" ] 
    # then
    echo "코퍼스 파일: ${file}" 
    echo "OUTPUT_DIR: ${OUTPUT_DIR}"
    echo "RESORCE_DIR: ${RESOURCE_DIR}"
    # 코퍼스 조각 -> tfrecord로 만드는 작업을 백그라운드에서 실행
    nohup \
    python3 bert-sentencepiece/create_pretraining_data.py \
    --input_file=$file \
    --output_file=gs://$OUTPUT_DIR/$TOKENIZER'_'$file_num.tfrecord \
    --vocab_file=gs://$RESOURCE_DIR/vocab.txt \
    --do_lower_case=True \
    --max_predictions_per_seq=20 \
    --max_seq_length=128 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --piece=sentence \
    --piece_model=$TOKENIZER'_'tok.model \
    --dupe_factor=5 > $TOKENIZER'_'$file_num'_'$file_num.log 2> $TOKENIZER'_'$file_num.err &

    file_num=$((file_num +1))
    echo $file_num
  
done

# log를 gcs로 전송
#gsutil mv *.err gs://$OUTPUT_DIR


