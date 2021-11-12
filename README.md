# acl_tokenization
acl

![mecab_fixed](https://github.com/taeheejeon22/konlpy-mecab-fixed)
<br>
![kortok](https://github.com/kakaobrain/kortok)


# tokenization stratagies
## *baselines*
### eojeol
- composed                WP only
- decomposed pure (nfd)   KR-BERT

### morpheme_orig
- (composed                kortok)
- (composed + decomposed pure)

### morpheme_fixed
- (composed + decomposed pure)


## *OUR*
### morpheme_orig
- (composed + grammar_symbol)
- (grammar_symbol + decomposed_lexical)

- (grammar_symbol + decomposed pure)
    - (composed + grammar_symbol + decomposed_grammatical)

### morpheme_fixed
- composed
- composed + grammar_symbol
- grammar_symbol + decomposed_lexical

- (grammar_symbol + decomposed pure)
    - (composed + grammar_symbol + decomposed_grammatical)
    


# 0. raw 코퍼스 만들기
parse_Wikiko_with_preprocessing_v0.py


# 1. 코퍼스 토큰화
./scripts/tokenization_v2_0.py
./pretrain_corpus/tokenized 에 저장

## wikiko (without dummy letter)
### baseline
```bash
python scripts/mecab_tokenization_v2.py --token_type=eojeol --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=composed
python scripts/mecab_tokenization_v2.py --token_type=eojeol --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure

python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_orig --decomposition_type=composed

python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_orig --decomposition_type=decomposed_pure
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure 
```

### our
```bash
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_orig --decomposition_type=composed --grammatical_symbol=⫸⭧
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_orig --decomposition_type=decomposed_lexical --grammatical_symbol=⫸⭧
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_orig --decomposition_type=decomposed_pure --grammatical_symbol=⫸⭧




python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=composed --grammatical_symbol=⫸⭧
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_lexical --grammatical_symbol=⫸⭧
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure --grammatical_symbol=⫸⭧
```




## namuwiki
분할 커맨드
```bash
split -d -l 5000000 namuwiki_20200302_with_preprocessing_v3_nn.txt namuwiki_20200302_with_preprocessing_v3_nn_
```

### baseline
```bash
python scripts/mecab_tokenization_v2.py --token_type=eojeol --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=composed
python scripts/mecab_tokenization_v2.py --token_type=eojeol --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure

python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_orig --decomposition_type=composed 
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_orig --decomposition_type=decomposed_pure
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=composed
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure 
```

### our
```bash
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_orig --decomposition_type=composed --grammatical_symbol="⭧" 
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_orig --decomposition_type=decomposed_lexical --grammatical_symbol="⭧"
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_orig --decomposition_type=decomposed_pure --grammatical_symbol="⭧"

python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=composed --grammatical_symbol="⭧" 
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_lexical --grammatical_symbol="⭧"
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure --grammatical_symbol="⭧"
```


메모리 부족할 경우
```bash
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_00 --tokenizer_type=mecab_fixed --decomposition_type=composed
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_01 --tokenizer_type=mecab_fixed --decomposition_type=composed
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_02 --tokenizer_type=mecab_fixed --decomposition_type=composed
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_03 --tokenizer_type=mecab_fixed --decomposition_type=composed
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_04 --tokenizer_type=mecab_fixed --decomposition_type=composed
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_05 --tokenizer_type=mecab_fixed --decomposition_type=composed

python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_00 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_01 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_02 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_03 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_04 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_05 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure

python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_00 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_morphological
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_01 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_morphological
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_02 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_morphological
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_03 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_morphological
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_04 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_morphological
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_05 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_morphological
```
orig decomposed pure 25분
orig decomposed morphological 15분



namuwiki (with dummy letter)
```bash
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_orig --decomposition_type=decomposed_pure --dummy_letter=⊸
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_orig --decomposition_type=decomposed_morphological --dummy_letter=⊸

python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure --dummy_letter=⊸
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_morphological --dummy_letter=⊸
```


메모리 부족할 경우
```bash
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_00 --tokenizer_type=mecab_orig --decomposition_type=decomposed_pure --dummy_letter=⊸
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_01 --tokenizer_type=mecab_orig --decomposition_type=decomposed_pure --dummy_letter=⊸
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_02 --tokenizer_type=mecab_orig --decomposition_type=decomposed_pure --dummy_letter=⊸
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_03 --tokenizer_type=mecab_orig --decomposition_type=decomposed_pure --dummy_letter=⊸
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_04 --tokenizer_type=mecab_orig --decomposition_type=decomposed_pure --dummy_letter=⊸
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_05 --tokenizer_type=mecab_orig --decomposition_type=decomposed_pure --dummy_letter=⊸

python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_orig --decomposition_type=decomposed_morphological --n_job=6 --dummy_letter=⊸

python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_00 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure --dummy_letter=⊸
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_01 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure --dummy_letter=⊸
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_02 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure --dummy_letter=⊸
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_03 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure --dummy_letter=⊸
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_04 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure --dummy_letter=⊸
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_05 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure --dummy_letter=⊸

python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_00 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_morphological --dummy_letter=⊸
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_01 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_morphological --dummy_letter=⊸
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_02 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_morphological --dummy_letter=⊸
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_03 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_morphological --dummy_letter=⊸
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_04 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_morphological --dummy_letter=⊸
python scripts/mecab_tokenization.py --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn_05 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_morphological --dummy_letter=⊸
```
orig decomposed morphological 20:12


- 도저히 안 되면 tokenization_v4.0.py 이용해서 할 것. 128G로도 메모리 터짐.

* namuwiki fixed 모든 세팅
* namuwiki (with dummy letter) orig decompsed pure 







# 2. Sentencepiece 학습

[comment]: <> (## mecab 토큰화)

[comment]: <> (./build_vocab/build_mecab_vocab_our.py &#40;자동화 위해 코드 수정 필요&#41;)

[comment]: <> (```bash)

[comment]: <> (python build_vocab/build_mecab_vocab_our.py --vocab_size=64000)

[comment]: <> (```)


## wordpiece

### baseline
```bash
python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F/wikiko_20210901_eojeol_mecab_fixed/composed --vocab_size=64000 --token_type=eojeol --tokenizer_type=mecab_fixed --composition_type=composed
python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F/wikiko_20210901_eojeol_mecab_fixed/decomposed_pure --vocab_size=64000 --token_type=eojeol --tokenizer_type=mecab_fixed --composition_type=decomposed_pure

python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F/wikiko_20210901_morpheme_mecab_orig/composed --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_orig --composition_type=composed



python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_orig --composition_type=composed
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_orig --composition_type=decomposed_pure
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_fixed --composition_type=composed
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_fixed --composition_type=decomposed_pure
```

### our
```bash
python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F/wikiko_20210901_morpheme_mecab_fixed/composed --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_fixed --composition_type=composed

```







## sentencepiece
./build_vocab/train_sentencepiece.py
### baseline
```bash
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=eojeol --tokenizer_type=mecab_fixed --composition_type=composed
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=eojeol --tokenizer_type=mecab_fixed --composition_type=decomposed_pure

python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_orig --composition_type=composed
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_orig --composition_type=decomposed_pure
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_fixed --composition_type=composed
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_fixed --composition_type=decomposed_pure
```

### our
```bash
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_T --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_orig --composition_type=composed
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_T --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_orig --composition_type=decomposed_lexical
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_T --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_orig --composition_type=decomposed_pure

python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_T --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_fixed --composition_type=composed
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_T --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_fixed --composition_type=decomposed_lexical
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_T --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_fixed --composition_type=decomposed_pure
```





# 3. make BERT files 
```buildoutcfg
python scripts/make_bert_files.py --root_path=output_sp/ --vocab_size=64000
```
실행 후 파일들 resources로 옮기기. 수동으로.


# 4. pretrain BERT
파일 분할 by size (https://stackoverflow.com/questions/17592725/get-file-size-and-split-the-file-based-on-size)

namuwiki (without dummy letter) split
```bash
split -d -l 8000000 namuwiki_20200302_none_composed.txt namuwiki_20200302_none_composed_

split -d -l 6000000 namuwiki_20200302_mecab_orig_composed.txt namuwiki_20200302_mecab_orig_composed_
split -d -l 3300000 namuwiki_20200302_mecab_orig_decomposed_pure.txt namuwiki_20200302_mecab_orig_decomposed_pure_
split -d -l 4000000 namuwiki_20200302_mecab_orig_decomposed_morphological.txt namuwiki_20200302_mecab_orig_decomposed_morphological_

split -d -l 6000000 namuwiki_20200302_tokenized_mecab_fixed_composed.txt namuwiki_20200302_tokenized_mecab_fixed_composed_
split -d -l 3000000 namuwiki_20200302_tokenized_mecab_fixed_decomposed_pure.txt namuwiki_20200302_tokenized_mecab_fixed_decomposed_pure_
split -d -l 5000000 namuwiki_20200302_tokenized_mecab_fixed_decomposed_morphological.txt namuwiki_20200302_tokenized_mecab_fixed_decomposed_morphological_

```

namuwiki (with dummy letter) split
```bash
split -d -l 2800000 namuwiki_20200302_tokenized_mecab_orig_decomposed_pure.txt namuwiki_20200302_tokenized_mecab_orig_decomposed_pure_
split -d -l 4000000 namuwiki_20200302_mecab_orig_decomposed_morphological.txt namuwiki_20200302_mecab_orig_decomposed_morphological_


split -d -l 2700000 namuwiki_20200302_mecab_fixed_decomposed_pure.txt namuwiki_20200302_mecab_fixed_decomposed_pure_
split -d -l 4300000 namuwiki_20200302_mecab_fixed_decomposed_morphological.txt namuwiki_20200302_mecab_fixed_decomposed_morphological_

```





## input 
- tokenized corpus:
- tok.vocab: ./resources/xx/
- bert_config.json: ./resources/xx/
