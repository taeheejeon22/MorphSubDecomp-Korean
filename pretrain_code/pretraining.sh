#!/bin/bash
# vocab.txt, bert_config.json이 있는 directory에서 실행.
# git clone https://github.com/google-research/bert.git

# 완성된 프리 트레인 모델은 /tmp/pretraining_output 에 저장됨.
# 현재 파라미터는 테스트용

set -e

python create_pretraining_data.py \
  --input_file=./wiki_00.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=./vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5





python run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=/tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=./bert_config.json \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
