# acl_tokenization
acl

![mecab_fixed](https://github.com/taeheejeon22/konlpy-mecab-fixed)
<br>
![kortok](https://github.com/kakaobrain/kortok)


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


# 3. make BERT files 
```buildoutcfg
python scripts/make_bert_files.py --root_path=resources/with_dummy_letter_v2/ --vocab_size=64000 
```


# 4. pretrain BERT
파일 분할 by size (https://stackoverflow.com/questions/17592725/get-file-size-and-split-the-file-based-on-size)
```bash
split -d -b2G namuwiki_20200302_tokenized_mecab_orig_decomposed_pure_all.txt ./split/namuwiki_20200302_tokenized_mecab_orig_decomposed_pure_

split -d -b2G namuwiki_20200302_tokenized_mecab_fixed_decomposed_pure_all.txt ./split/namuwiki_20200302_tokenized_mecab_fixed_decomposed_pure_

```


## input 
- tokenized corpus:
- tok.vocab: ./resources/xx/
- bert_config.json: ./resources/xx/
