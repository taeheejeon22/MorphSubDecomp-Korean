### 기존 resources의 파일의 이름을 다음과 같이 수정해야 함.
bert_model: 원래 모델명 -> "pytorch_model.bin"
tokenizer: "tok.model" -> "bert_tokenizer.json"
config.json: "bert_config.json" -> "config.json"

### 기존 resources에 더해 추가적으로 필요한 파일
special_tokens_map.json
tokenizer_config.json

### 내용 수정
"bert_config.json"에   "model_type": "bert", 추가.
