#!/bin/bash

# pretraining_code_stage1.sh 를 실행시켜서 만든 tfrecord 파일들로 
# ckpt 파일을 만드는 작업

# 필요한 도구 받기

#pip install tensorflow==1.14
#pip install -U gast==0.2.2
#git clone https://github.com/google-research/bert.git

# tokenizer를 사용자로부터 입력 받기
echo "아래의 토크나이저 중에서 사용할 토크나이저를 입력하세요. "
echo "eojeol_fixed_composed"
echo "eojeol_fixed_decomposed_pure"
echo "morpheme_orig_composed    morpheme_orig_decomposed_pure    morpheme_orig_decomposed_morphological"
echo "morpheme_fixed_composed    morpheme_fixed_decomposed_pure    morpheme_fixed_decomposed_morphological"

echo -e "tokenizer: " 
read TOKENIZER
echo "tokenizer == $TOKENIZER"

# tfrecord dir, resource_dir, model_dir 입력 받기
echo "tfrecord_dir을 입력하세요."
echo -e "tfrocord_dir: "
read TFRECORD_DIR
echo "tfrecord_dir == $TFRECORD_DIR"

echo "resource_dir을 입력하세요. "
echo -e "resource_dir: "
read RESOURCE_DIR
echo "resource_dir == $RESOURCE_DIR"

echo "bert_model_dir을 입력하세요."
echo -e "bert_model_dir: "
read MODEL_DIR
echo "bert_model_dir == $MODEL_DIR"

# tpu name, region 입력 받기
echo "tpu name을 입력하세요."
echo -e "tpu_name: "
read TPU_NAME
echo "TPU_NAME == $TPU_NAME"

echo "region을 입력하세요."
echo -e "region: "
read REGION
echo "region == $REGION"

# init_checkpoints의 여부
echo "init_checkpoints 쓰면 T, 안 쓰면 F"
echo -e "T or F: "
read INIT
echo "init == $INIT"
if [[ $INIT == "T" ]]; then
    echo -e "init_checkpoints를 입력하세요: "
    read INIT_CHECKPOINTS
    echo "init_checkpoints == $INIT_CHECKPOINTS"
else
    echo "pass"
fi



echo tfrecode_dir == $TFRECORD_DIR
echo resource_dir == $RESOURCE_DIR
echo model_dir == $MODEL_DIR


# run_pretraining.py 실행 (백그라운드)

if [[ $INIT == "F" ]]; then

    nohup \
    python3 bert/run_pretraining.py \
    --input_file=gs://$TFRECORD_DIR/*.tfrecord \
    --output_dir=gs://$MODEL_DIR \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=gs://$RESOURCE_DIR/bert_config.json \
    --train_batch_size=1024 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --num_train_steps=1000000 \
    --num_warmup_steps=10000 \
    --learning_rate=5e-5 \
    --save_checkpoints_steps=20000 \
    --use_tpu=True \
    --do_lower_case=False \
    --tpu_name=$TPU_NAME \
    --tpu_zone=$REGION \
    --gcp_project=smooth-loop-327807 \
    --num_tpu_cores=8 > ${TOKENIZER}.log 2>&1 &
else
    nohup \
    python3 bert/run_pretraining.py \
    --input_file=gs://$TFRECORD_DIR/*.tfrecord \
    --output_dir=gs://$MODEL_DIR \
    --do_train=True \
    --do_eval=True \
    --bert_config_file=gs://$RESOURCE_DIR/bert_config.json \
    --train_batch_size=1024 \
    --max_seq_length=128 \
    --max_predictions_per_seq=20 \
    --num_train_steps=1000000 \
    --num_warmup_steps=10000 \
    --learning_rate=5e-5 \
    --save_checkpoints_steps=20000 \
    --init_checkpoints=$INIT_CHECKPOINTS \
    --use_tpu=True \
    --do_lower_case=False \
    --tpu_name=$TPU_NAME \
    --tpu_zone=$REGION \
    --gcp_project=smooth-loop-327807 \
    --num_tpu_cores=8 > ${TOKENIZER}.log 2>&1 &
fi


# save command log
echo $TOKENIZER' ### '$TFRECORD_DIR'  ### '$RESOURCE_DIR' ### '$MODEL_DIR &> ${TOKENIZER}'_'command.log   


