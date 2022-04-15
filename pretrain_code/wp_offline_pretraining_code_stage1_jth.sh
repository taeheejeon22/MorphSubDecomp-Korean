#!/bin/bash

# 코퍼스, vocab.txt를 바탕으로 tfrecord 파일을 만드는 코드
# 각 토크나이저로 토큰화한 코퍼스 파일들을 모든 파일들을 차례로 불러온 뒤,
# 각 파일에 대한 tfrecord 파일을 만듭니다.


set -e

# 필요한 도구 받기
#pip install tensorflow==1.14
#pip install -U gast==0.2.2
#git clone https://github.com/google-research/bert.git

# tokenizer를 사용자로부터 입력 받기
echo "아래의 토크나이저 중에서 사용할 토크나이저를 입력하세요. "
echo "eojeol_fixed_composed"
echo "eojeol_fixed_pure"
echo "morpheme_orig_composed    morpheme_orig_decomposed_pure"
echo "morpheme_fixed_composed    morpheme_fixed_decomposed_pure"

echo "morpheme_fixed_composed_T    morpheme_fixed_decomposed_pure_T"
echo "morpheme_fixed_decomposed_lexical_T    morpheme_fixed_decomposed_grammatical_T"
echo "morpheme_fixed_decomposed_lexical_F"
echo "morpheme_fixed_decomposed_grammatical_F"

### jth 2022-03-09
echo "LG_mecab_fixed_composed_F"
echo "LG_mecab_fixed_decomposed_pure_F"
echo "LG_mecab_fixed_decomposed_lexical_F"
echo "LG_mecab_fixed_decomposed_grammatical_F"
###


echo -e "tokenizer: " 
read TOKENIZER
echo "tokenizer == $TOKENIZER"

echo "corpus의 디렉토리를 입력하세요."
echo -e "corpus_dir: "
read CORPUS_DIR
echo "corpus_dir == $CORPUS_DIR"

echo "resource의 디렉토리를 입력하세요. "
echo -e "resource_dir: "
read RESOURCE_DIR
echo "resource_dir == $RESOURCE_DIR"

# 입력 받은 tokenizer, corpus의 output_dir

#OUTPUT_DIR=`echo ${CORPUS_DIR//"tokenized_GCP"/"tfrecord"}`
OUTPUT_DIR=../corpus/tfrecord


if [ ! -d ${OUTPUT_DIR}/${TOKENIZER} ]; then
    echo "OUTPUT_DIR making..."
    mkdir -p ${OUTPUT_DIR}/${TOKENIZER}
    echo "OUTPUT_DIR making... Done!"
fi


# 각 코퍼스 파일에 대해서 tfrecord 만들기

file_num=0

for file in `ls $CORPUS_DIR/*`
do
    file_name=`basename ${file} .txt`
    echo "코퍼스 파일: ${file_name}" 
    echo "OUTPUT_DIR: ${OUTPUT_DIR}"
    echo "RESORCE_DIR: ${RESOURCE_DIR}"

    nohup \
    python bert/create_pretraining_data.py \
    --input_file=${file} \
    --output_file=${OUTPUT_DIR}/${TOKENIZER}/${file_name}.tfrecord \
    --vocab_file=${RESOURCE_DIR}/vocab.txt \
    --do_lower_case=False \
    --max_predictions_per_seq=20 \
    --max_seq_length=128 \
    --masked_lm_prob=0.15 \
    --random_seed=12345 \
    --dupe_factor=5 > ${TOKENIZER}'_'${file_name}.log 2>&1 &

    echo $TOKENIZER' ### '$file_name' ### '$CORPUS_DIR' ### '$RESOURCE_DIR' ### '${OUTPUT_DIR}/${file_name}.tfrecord &> ${file_name}'_'command.log      

done

