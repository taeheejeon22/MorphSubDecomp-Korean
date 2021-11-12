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

#OUTPUT_DIR=`echo ${CORPUS_DIR//"tokenized_GCP"/"tfrecord"}`
OUTPUT_DIR=/home/jth/Desktop/acl_tokenization/corpus/tfrecord




# 각 코퍼스 파일에 대해서 tfrecord 만들기

file_num=0

for file in `ls $CORPUS_DIR/*`
do
    file_name=`basename ${file} .txt`
    echo "코퍼스 파일: ${file_name}" 
    echo "OUTPUT_DIR: ${OUTPUT_DIR}"
    echo "RESORCE_DIR: ${RESOURCE_DIR}"
    # 코퍼스 조각 -> tfrecord로 만드는 작업을 백그라운드에서 실행
    nohup \
    python3 bert/create_pretraining_data.py \
    --input_file=${file} \
    --output_file=${OUTPUT_DIR}/${TOKENIZER}/${file_name}.tfrecord \
    --vocab_file=${RESOURCE_DIR}/vocab.txt \
    --do_lower_case=True \
    --max_predictions_per_seq=20 \
    --max_seq_length=128 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=5 > ${TOKENIZER}'_'${file_name}.log 2>&1 &

    # save command log
    echo $TOKENIZER' ### '$file_name' ### '$CORPUS_DIR' ### '$RESOURCE_DIR' ### '${OUTPUT_DIR}/${file_name}.tfrecord &> ${file_name}'_'command.log   

done

# /home/jth/Desktop/acl_tokenization/corpus/tokenized/without_dummy_letter/namuwiki_20200302_eojeol_mecab_fixed/composed_nfd

# /home/jth/Desktop/acl_tokenization/resources/v6_without_dummy_letter/eojeol_mecab_fixed_composed_wp-64k

# /home/jth/Desktop/acl_tokenization/corpus/tokenized/without_dummy_letter/namuwiki_20200302_eojeol_mecab_fixed/decomposed_pure_nfd/1st

# /home/jth/Desktop/acl_tokenization/resources/v6_without_dummy_letter/eojeol_mecab_fixed_decomposed_pure_wp-64k

# /home/jth/Desktop/acl_tokenization/corpus/tokenized/without_dummy_letter/namuwiki_20200302_eojeol_mecab_fixed/decomposed_pure_nfd/2nd



# /home/jth/Desktop/acl_tokenization/corpus/fake