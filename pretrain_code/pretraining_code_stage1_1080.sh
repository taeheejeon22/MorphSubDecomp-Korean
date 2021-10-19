#!/bin/bash

# 코퍼스, vocab.txt를 바탕으로 tfrecord 파일을 만드는 코드
# 각 토크나이저로 토큰화한 코퍼스 파일들이 있는 폴더에 있는 모든 파일들을 차례로 불러온 뒤,
# 각 파일에 대한 tfrecord 파일을 만든다.


set -e

# 필요한 도구 받기
#pip install tensorflow==1.14
pip install -U gast==0.2.2
git clone https://github.com/google-research/bert.git



# google storage 주소
$GCS = gs://kist_bert


# tokenizer를 사용자로부터 입력 받기
echo "아래의 토크나이저 중에서 사용할 토크나이저를 입력하세요. "
echo "none_composed"
echo "orig_composed    orig_decomposed_pure    orig_deocomposed_morphological"
echo "fixed_composed    fixed_decomposed_pure    fixed_decomposed_morphological"

echo -e "tokenizer:  c " 
read TOKENIZER
echo "tokenizer == $TOKENIZER"


# 입력 받은 tokenizer의 wiki_corpus_dir, namuwiki_corpus_dir, resource_dir, output_dir을 정하기

# none_composed
if [ $TOKENIZER == none_composed ]; then
    $wiki_corpus_dir=$GCS/tokenized_GCP/with_dummy_letter/wiki
    $namuwiki_corpus_dir=$GCS/tokenized_GCP/with_dummy_letter/namuwiki/$TOKENIZER
    $resource_dir=$GCS/resources/with_dummy_letter_v2/sp-64k
    $output_dir=$GCS/tfrecord/v2/$TOKENIZER

#orig_composed
elif [ $TOKENIZER == orig_composed ]; then
    $wiki_corpus_dir=$GCS/tokenized_GCP/with_dummy_letter/wiki
    $namuwiki_corpus_dir=$GCS/tokenized_GCP/with_dummy_letter/namuwiki/$TOKENIZER
    $resource_dir=$GCS/resources/with_dummy_letter_v2/mecab_$TOKENIZER'_sp-64k'/
    $output_dir=$GCS/tfrecord/v2/$TOKENIZER

# orig_decomposed_pure
elif [ $TOKENIZER == orig_decomposed_pure ]; then
    $wiki_corpus_dir=$GCS/tokenized_GCP/with_dummy_letter/wiki
    $namuwiki_corpus_dir=$GCS/tokenized_GCP/with_dummy_letter/namuwiki/$TOKENIZER
    $resource_dir=$GCS/resources/with_dummy_letter_v2/mecab_$TOKENIZER'_sp-64k'
    $output_dir=$GCS/tfrecord/v2/$TOKENIZER

# orig_decomposed_morphological
elif [ $TOKENIZER == orig_deocomposed_morphological ]; then
    $wiki_corpus_dir=$GCS/tokenized_GCP/with_dummy_letter/wiki
    $namuwiki_corpus_dir=$GCS/tokenized_GCP/with_dummy_letter/namuwiki/$TOKENIZER
    $resource_dir=$GCS/resources/with_dummy_letter_v2/mecab_$TOKENIZER'_sp-64k'
    $output_dir=$GCS/tfrecord/v2/$TOKENIZER

#fixed_composed
elif [ $TOKENIZER == fixed_composed ] ; then
    $wiki_corpus_dir=$GCS/tokenized_GCP/with_dummy_letter/wiki
    $namuwiki_corpus_dir=$GCS/tokenized_GCP/with_dummy_letter/namuwiki/$TOKENIZER
    $resource_dir=$GCS/resources/with_dummy_letter_v2/mecab_$TOKENIZER'_sp-64k'
    $output_dir=$GCS/tfrecord/v2/$TOKENIZER

#fixed_decomposed_pure
elif [ $TOKENIZER == fixed_decomposed_pure ] ; then
    $wiki_corpus_dir=$GCS/tokenized_GCP/with_dummy_letter/wiki
    $namuwiki_corpus_dir=$GCS/tokenized_GCP/with_dummy_letter/namuwiki/$TOKENIZER
    $resource_dir=$GCS/resources/with_dummy_letter_v2/mecab_$TOKENIZER'_sp-64k'
    $output_dir=$GCS/tfrecord/v2/$TOKENIZER

#fixed_decomposed_morphological
elif [ $TOKENIZER == fixed_decomposed_morphological ] ; then
    $wiki_corpus_dir=$GCS/tokenized_GCP/with_dummy_letter/wiki
    $namuwiki_corpus_dir=$GCS/tokenized_GCP/with_dummy_letter/namuwiki/$TOKENIZER
    $resource_dir=$GCS/resources/with_dummy_letter_v2/mecab_$TOKENIZER'_sp-64k'
    $output_dir=$GCS/tfrecord/v2/$TOKENIZER

fi


# 각 코퍼스 파일에 대해서 tfrecord 만들기

# wiki
corpus_files=`ls $wiki_corpus_dir | grep *\.txt$`
file_count=`ls $wiki_corpus_dir -l | grep ^-.*\.txt$ | wc -l`

echo corpus files: $corpus_files
echo 코퍼스 파일 수: $file_count


for file in $corpus_files
do
    output_filename=`echo $file | cut -d'.' -f1`
    python3 bert/create_pretraining_data.py \
      --input_file=$wiki_corpus_dir/$file \
      --output_file=output_dir/$output_filename.tfrecord \
      --vocab_file=$resource_dir/vocab.txt \
      --do_lower_case=True \
      --max_predictions_per_seq=20 \
      --max_seq_length=128 \
      --masked_lm_prob=0.15 \
      --random_seed=12345 \
      --dupe_factor=5 2>&1 |tee -a $file'_'log.txt
done


# namuwiki

corpus_files=`ls $namuwiki_corpus_dir | grep *\.txt$`
file_count=`ls $namuwiki_corpus_dir -l | grep ^-.*\.txt$ | wc -l`

echo corpus files: $corpus_files
echo 코퍼스 파일 수: $file_count


for file in $corpus_files
do

    output_filename=`echo $file | cut -d'.' -f1`
    python3 bert/create_pretraining_data.py \
      --input_file=$namuwiki_corpus_dir/$file \
      --output_file=output_dir/$output_filename.tfrecord \
      --vocab_file=$resource_dir/vocab.txt \
      --do_lower_case=True \
      --max_predictions_per_seq=20 \
      --max_seq_length=128 \
      --masked_lm_prob=0.15 \
      --random_seed=12345 \
      --dupe_factor=5 2>&1 |tee -a $file'_'log.txt
done

