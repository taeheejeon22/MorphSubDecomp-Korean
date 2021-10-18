#!/bin/bash

# pretraining_code_stage1.sh 를 실행시켜서 만든 tfrecord 파일들로 
# ckpt 파일을 만드는 작업

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


# tpu name, region 입력 받기
echo "tpu name을 입력하세요."
echo -e "tpu_name: c "
read TPU_NAME
echo "TPU_NAME == $TPU_NAME"

echo "region을 입력하세요."
echo -e "region: c "
read REGION
echo "region == $REGION"

# 입력 받은 tokenizer의 tfrecord_dir, resource_dir, output_dir, model_dir을 정하기

# none_composed
if [ $TOKENIZER == none_composed ]; then
    $tfrecord_dir=$GCS/tfrecord/v2/$TOKENIZER
    $resource_dir=$GCS/resources/with_dummy_letter_v2/sp-64k\
    $model_dir=$GCS/bert_model/v2/$TOKENIZER

#orig_composed
elif [ $TOKENIZER == orig_composed ]; then
    $tfrecord_dir=$GCS/tfrecord/v2/$TOKENIZER
    $resource_dir=$GCS/resources/with_dummy_letter_v2/mecab_$TOKENIZER'_sp-64k'/
    $model_dir=$GCS/bert_model/v2/$TOKENIZER

# orig_decomposed_pure
elif [ $TOKENIZER == orig_decomposed_pure ]; then
    $tfrecord_dir=$GCS/tfrecord/v2/$TOKENIZER
    $resource_dir=$GCS/resources/with_dummy_letter_v2/mecab_$TOKENIZER'_sp-64k'
    $model_dir=$GCS/bert_model/v2/$TOKENIZER

# orig_decomposed_morphological
elif [ $TOKENIZER == orig_deocomposed_morphological ]; then
    $tfrecord_dir=$GCS/tfrecord/v2/$TOKENIZER
    $resource_dir=$GCS/resources/with_dummy_letter_v2/mecab_$TOKENIZER'_sp-64k'
    $model_dir=$GCS/bert_model/v2/$TOKENIZER

#fixed_composed
elif [ $TOKENIZER == fixed_composed ] ; then
    $tfrecord_dir=$GCS/tfrecord/v2/$TOKENIZER
    $resource_dir=$GCS/resources/with_dummy_letter_v2/mecab_$TOKENIZER'_sp-64k'
    $model_dir=$GCS/bert_model/v2/$TOKENIZER

#fixed_decomposed_pure
elif [ $TOKENIZER == fixed_decomposed_pure ] ; then
    $tfrecord_dir=$GCS/tfrecord/v2/$TOKENIZER
    $resource_dir=$GCS/resources/with_dummy_letter_v2/mecab_$TOKENIZER'_sp-64k'
    $model_dir=$GCS/bert_model/v2/$TOKENIZER

#fixed_decomposed_morphological
elif [ $TOKENIZER == fixed_decomposed_morphological ] ; then
    $tfrecord_dir=$GCS/tfrecord/v2/$TOKENIZER
    $resource_dir=$GCS/resources/with_dummy_letter_v2/mecab_$TOKENIZER'_sp-64k'
    $model_dir=$GCS/bert_model/v2/$TOKENIZER
fi



# run_pretraining.py 실행

python3 bert/run_pretraining.py \
--input_file=$tfrecord_dir/*.tfrecord \
--output_dir=$model_dir \
--do_train=True \
--do_eval=True \
--bert_config_file=$resource_dir/bert_config.json \
--train_batch_size=1024 \
--max_seq_length=128 \
--max_predictions_per_seq=20 \
--num_train_steps=1000000 \
--num_warmup_steps=10000 \
--learning_rate=5e-5 \
--save_checkpoints_steps=10000 \
--use_tpu=True \
--tpu_name=$TPU_NAME \
--tpu_zone=$REGION \
--gcp_project=smooth-loop-327807 \
--num_tpu_cores=8 2>&1 |tee -a $TOKENIZER'_'pretrain_log.txt