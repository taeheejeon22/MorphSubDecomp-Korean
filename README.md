# acl_tokenization
acl

![mecab_fixed](https://github.com/taeheejeon22/konlpy-mecab-fixed)
<br>
![kortok](https://github.com/kakaobrain/kortok)


# 0. raw 코퍼스 만들기
parse_Wikiko_with_preprocessing_v0.py


# 1. 코퍼스 토큰화
./scripts/tokenization_v2_0.py
./pretrain_corpus/tokenized 에 저장

wikiko
```bash
python scripts/mecab_tokenization.py --tokenizer_type=mecab_orig --decomposition_type=composed
python scripts/mecab_tokenization.py --tokenizer_type=mecab_orig --decomposition_type=decomposed_pure
python scripts/mecab_tokenization.py --tokenizer_type=mecab_orig --decomposition_type=decomposed_morphological
python scripts/mecab_tokenization.py --tokenizer_type=mecab_fixed --decomposition_type=composed
python scripts/mecab_tokenization.py --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure
python scripts/mecab_tokenization.py --tokenizer_type=mecab_fixed --decomposition_type=decomposed_morphological

```

namuwiki
```bash
python scripts/mecab_tokenization.py --tokenizer_type=none

python scripts/mecab_tokenization.py --tokenizer_type=mecab_orig --decomposition_type=composed
python scripts/mecab_tokenization.py --tokenizer_type=mecab_orig --decomposition_type=decomposed_pure --n_job=10
python scripts/mecab_tokenization.py --tokenizer_type=mecab_orig --decomposition_type=decomposed_morphological --n_job=10
python scripts/mecab_tokenization.py --tokenizer_type=mecab_fixed --decomposition_type=composed --n_job=4
python scripts/mecab_tokenization.py --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure --n_job=1
python scripts/mecab_tokenization.py --tokenizer_type=mecab_fixed --decomposition_type=decomposed_morphological --n_job=6
```








# 2. Sentencepiece 학습

[comment]: <> (## mecab 토큰화)

[comment]: <> (./build_vocab/build_mecab_vocab_our.py &#40;자동화 위해 코드 수정 필요&#41;)

[comment]: <> (```bash)

[comment]: <> (python build_vocab/build_mecab_vocab_our.py --vocab_size=64000)

[comment]: <> (```)


## sentencepiece
./build_vocab/train_sentencepiece.py
```bash
python build_vocab/train_sentencepiece.py --vocab_size=32000

python build_vocab/train_sentencepiece.py --vocab_size=32000 --tokenizer_type="mecab_orig" --composition_type="composed"

python build_vocab/train_sentencepiece.py --vocab_size=32000 --tokenizer_type="mecab_orig" --composition_type="decomposed_pure"

python build_vocab/train_sentencepiece.py --vocab_size=32000 --tokenizer_type="mecab_orig" --composition_type="decomposed_morphological"

python build_vocab/train_sentencepiece.py --vocab_size=32000 --tokenizer_type="mecab_fixed" --composition_type="composed"

python build_vocab/train_sentencepiece.py --vocab_size=32000 --tokenizer_type="mecab_fixed" --composition_type="decomposed_pure"

python build_vocab/train_sentencepiece.py --vocab_size=32000 --tokenizer_type="mecab_fixed" --composition_type="decomposed_morphological"

```


# 3. make BERT files 
```buildoutcfg
python scripts/make_bert_files.py --root_path=output_sp/ --vocab_size=32000 
```
실행 후 파일들 resources로 옮기기. 수동으로.


# 4. pretrain BERT
파일 분할 by size (https://stackoverflow.com/questions/17592725/get-file-size-and-split-the-file-based-on-size)

namuwiki split
```bash
split -d -l 8000000 namuwiki_20200302_none_composed.txt namuwiki_20200302_none_composed_

split -d -l 6000000 namuwiki_20200302_mecab_orig_composed.txt namuwiki_20200302_mecab_orig_composed_
split -d -l 3300000 namuwiki_20200302_mecab_orig_decomposed_pure.txt namuwiki_20200302_mecab_orig_decomposed_pure_
split -d -l 4000000 namuwiki_20200302_mecab_orig_decomposed_morphological.txt namuwiki_20200302_mecab_orig_decomposed_morphological_


```




## input 
- tokenized corpus:
- tok.vocab: ./resources/xx/
- bert_config.json: ./resources/xx/
