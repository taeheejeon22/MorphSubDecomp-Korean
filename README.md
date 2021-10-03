# acl_tokenization
acl

![mecab_fixed](https://github.com/taeheejeon22/konlpy-mecab-fixed)


# 0. raw 코퍼스 만들기

# 1. 코퍼스 토큰화
./scripts/tokenization_v2_0.py

./pretrain_corpus/tokenized 에 저장


# 2. Sentencepiece 학습

[comment]: <> (## mecab 토큰화)

[comment]: <> (./build_vocab/build_mecab_vocab_our.py &#40;자동화 위해 코드 수정 필요&#41;)

[comment]: <> (```bash)

[comment]: <> (python build_vocab/build_mecab_vocab_our.py --vocab_size=64000)

[comment]: <> (```)


## sentencepiece
./build_vocab/train_sentencepiece.py
```bash
python build_vocab/train_sentencepiece.py --vocab_size=64000

python build_vocab/train_sentencepiece.py --vocab_size=64000 --tokenizer_type="mecab_orig" --composition_type="composed"

python build_vocab/train_sentencepiece.py --vocab_size=64000 --tokenizer_type="mecab_orig" --composition_type="decomposed_pure"

python build_vocab/train_sentencepiece.py --vocab_size=64000 --tokenizer_type="mecab_orig" --composition_type="decomposed_morphological"

python build_vocab/train_sentencepiece.py --vocab_size=64000 --tokenizer_type="mecab_fixed" --composition_type="composed"

python build_vocab/train_sentencepiece.py --vocab_size=64000 --tokenizer_type="mecab_fixed" --composition_type="decomposed_pure"

python build_vocab/train_sentencepiece.py --vocab_size=64000 --tokenizer_type="mecab_fixed" --composition_type="decomposed_morphological"

```


# 3. pretrain BERT
## input 
- tokenized corpus:
- tok.vocab: ./resources/xx/
- bert_config.json: ./resources/xx/
   kortok에 생성 코드가 없어서 그냥 복붙. 자동 생성 코드 추가해야 함.