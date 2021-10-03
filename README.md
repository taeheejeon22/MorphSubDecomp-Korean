# acl_tokenization
acl

![mecab_fixed](https://github.com/taeheejeon22/konlpy-mecab-fixed)


# 0. raw 코퍼스 만들기

# 1. 코퍼스 토큰화
./scripts/tokenization_v2_0.py

./pretrain_corpus/tokenized 에 저장


# 2. Sentencepiece 학습
## mecab 토큰화
./build_vocab/build_mecab_vocab_our.py

## sentencepiece
./build_vocab/train_sentencepiece.py


# 3. pretrain BERT
## input 
- tokenized corpus:
- tok.vocab: ./resources/xx/
- bert_config.json: ./resources/xx/
   kortok에 생성 코드가 없어서 그냥 복붙