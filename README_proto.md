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
- composed                kortok
- decomposed pure


## *OUR*
### morpheme_fixed
- composed
        
- composed + grammar_symbol
- grammar_symbol + decomposed_lexical   # 최고일 것으로 기대
- grammar_symbol + decomposed pure
    
    

## additional
### morpheme_fixed
- decomposed pure
- decomposed_lexical
- decomposed_grammatical
- grammar_symbol + decomposed_grammatical



# 0. raw 코퍼스 만들기
parse_Wikiko_with_preprocessing_v0.py


# 1. 코퍼스 토큰화
./scripts/tokenization_v2_0.py
./pretrain_corpus/tokenized 에 저장

## *baselines*
### wikiko 
```bash
python scripts/mecab_tokenization_v2.py --token_type=eojeol --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=composed
python scripts/mecab_tokenization_v2.py --token_type=eojeol --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure --nfd

python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_orig --decomposition_type=composed
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_orig --decomposition_type=decomposed_pure --nfd
```

### namuwiki
```bash
python scripts/mecab_tokenization_v2.py --token_type=eojeol --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=composed
python scripts/mecab_tokenization_v2.py --token_type=eojeol --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure --nfd

python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_orig --decomposition_type=composed
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_orig --decomposition_type=decomposed_pure --nfd
```

## *OUR*
### wikiko
```bash
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=composed

python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=composed --nfd --grammatical_symbol=⫸⭧
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_lexical --nfd --grammatical_symbol=⫸⭧
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure --nfd --grammatical_symbol=⫸⭧

# LG
python scripts/mecab_tokenization_v2_1.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=composed --lexical_grammatical
python scripts/mecab_tokenization_v2_1.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure --lexical_grammatical --nfd
python scripts/mecab_tokenization_v2_1.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_lexical --lexical_grammatical --nfd
python scripts/mecab_tokenization_v2_1.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_grammatical --lexical_grammatical --nfd
```

### namuwiki
```bash
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=composed

python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=composed --nfd --grammatical_symbol=⫸⭧
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_lexical --nfd --grammatical_symbol=⫸⭧
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure --nfd --grammatical_symbol=⫸⭧

# LG
python scripts/mecab_tokenization_v2_1.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=composed --lexical_grammatical --nfd
python scripts/mecab_tokenization_v2_1.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure --lexical_grammatical --nfd
python scripts/mecab_tokenization_v2_1.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_lexical --lexical_grammatical --nfd
python scripts/mecab_tokenization_v2_1.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_grammatical --lexical_grammatical --nfd
```

## additional
### wikiko
```bash
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure --nfd 
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_lexical --nfd 
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_grammatical --nfd 

python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_grammatical --nfd --grammatical_symbol=⫸⭧


```

### namuwiki
```bash
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_pure --nfd 
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_lexical --nfd 
python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_grammatical --nfd 

python scripts/mecab_tokenization_v2.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_v3_nn.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_grammatical --nfd --grammatical_symbol=⫸⭧


```

## namuwiki
분할 커맨드
```bash
split -d -l 8000000 namuwiki_20200302_with_preprocessing_v3_nn.txt namuwiki_20200302_with_preprocessing_v3_nn_
```

- 도저히 안 되면 tokenization_v4.0.py 이용해서 할 것. 128G로도 메모리 터짐.






# 2. Wordpiece, Sentencepiece 학습
[comment]: <> (## mecab 토큰화)

[comment]: <> (./build_vocab/build_mecab_vocab_our.py &#40;자동화 위해 코드 수정 필요&#41;)

[comment]: <> (```bash)

[comment]: <> (python build_vocab/build_mecab_vocab_our.py --vocab_size=64000)

[comment]: <> (```)


## Wordpiece
./build_vocab/train_wordsentencepiece.py
### *baselines*
```bash
python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F/eojeol_mecab_fixed/composed --vocab_size=64000
python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F/eojeol_mecab_fixed/decomposed_pure --vocab_size=64000

python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F/morpheme_mecab_orig/composed --vocab_size=64000
python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F/morpheme_mecab_orig/decomposed_pure --vocab_size=64000
```
- space_F_dummy_F_grammatical_F mecab_tokenizeation_v2.py에서만 있음. v2_1에서는 없어짐.

### *OUR*
```bash
python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F/morpheme_mecab_fixed/composed --vocab_size=64000
python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F/morpheme_mecab_fixed/decomposed_lexical --vocab_size=64000
python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F/morpheme_mecab_fixed/decomposed_pure --vocab_size=64000

python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F/morpheme_mecab_fixed/decomposed_grammatical --vocab_size=64000

# LG
python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/LG_mecab_fixed/composed_dummy_F --vocab_size=64000
python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/LG_mecab_fixed/decomposed_pure_dummy_F --vocab_size=64000
python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/LG_mecab_fixed/decomposed_lexical_dummy_F --vocab_size=64000

python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/LG_mecab_fixed/decomposed_grammatical_dummy_F --vocab_size=64000
```


## Sentencepiece
./build_vocab/train_sentencepiece.py
### *baselines*
```bash
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=eojeol --tokenizer_type=mecab_fixed --composition_type=composed
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=eojeol --tokenizer_type=mecab_fixed --composition_type=decomposed_pure

python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_orig --composition_type=composed
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_orig --composition_type=decomposed_pure
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_fixed --composition_type=composed
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_fixed --composition_type=decomposed_pure
```

### *OUR*
```bash
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_T --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_orig --composition_type=composed
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_T --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_orig --composition_type=decomposed_lexical
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_T --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_orig --composition_type=decomposed_pure

python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_T --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_fixed --composition_type=composed
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_T --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_fixed --composition_type=decomposed_lexical
python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_T --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_fixed --composition_type=decomposed_pure
```






# 3. make BERT files 
```bash
python scripts/make_bert_files.py --root_path=output_sp/ --vocab_size=64000 --model_max_length=128
```
실행 후 파일들 resources로 옮기기. 수동으로.



# 4. pretrain BERT
파일 분할 by size (https://stackoverflow.com/questions/17592725/get-file-size-and-split-the-file-based-on-size)


wiki (without dummy letter) split
```bash
# eojeol
split -d -l 2000000 namuwiki_20200302_none_composed.txt namuwiki_20200302_none_composed_
split -d -l 1500000 wikiko_20210901_eojeol_mecab_fixed_composed.txt wikiko_20210901_eojeol_mecab_fixed_composed_

split -d -l 1000000 namuwiki_20200302_eojeol_mecab_fixed_decomposed_pure_nfd.txt namuwiki_20200302_eojeol_mecab_fixed_decomposed_pure_nfd_
split -d -l 1000000 wikiko_20210901_eojeol_mecab_fixed_decomposed_pure.txt wikiko_20210901_eojeol_mecab_fixed_decomposed_pure_


# orig
split -d -l 1500000 namuwiki_20200302_morpheme_mecab_orig_composed.txt namuwiki_20200302_morpheme_mecab_orig_composed_
split -d -l 1500000 wikiko_20210901_morpheme_mecab_orig_composed.txt wikiko_20210901_morpheme_mecab_orig_composed_

split -d -l 800000 namuwiki_20200302_morpheme_mecab_orig_decomposed_pure_nfd.txt namuwiki_20200302_morpheme_mecab_orig_decomposed_pure_nfd_
split -d -l 600000 wikiko_20210901_morpheme_mecab_orig_decomposed_pure.txt wikiko_20210901_morpheme_mecab_orig_decomposed_pure_


# fixed
split -d -l 1200000 namuwiki_20200302_morpheme_mecab_fixed_composed.txt namuwiki_20200302_morpheme_mecab_fixed_composed_
split -d -l 1200000 wikiko_20210901_morpheme_mecab_fixed_composed.txt wikiko_20210901_morpheme_mecab_fixed_composed_

split -d -l 700000 namuwiki_20200302_morpheme_mecab_fixed_decomposed_pure.txt namuwiki_20200302_morpheme_mecab_fixed_decomposed_pure_
split -d -l 700000 wikiko_20210901_morpheme_mecab_fixed_decomposed_pure.txt wikiko_20210901_morpheme_mecab_fixed_decomposed_pure_

split -d -l 700000 namuwiki_20200302_morpheme_mecab_fixed_decomposed_lexical.txt namuwiki_20200302_morpheme_mecab_fixed_decomposed_lexical_
split -d -l 700000 wikiko_20210901_morpheme_mecab_fixed_decomposed_lexical.txt wikiko_20210901_morpheme_mecab_fixed_decomposed_lexical_


split -d -l 900000 namuwiki_20200302_morpheme_mecab_fixed_decomposed_grammatical.txt namuwiki_20200302_morpheme_mecab_fixed_decomposed_grammatical_
split -d -l 900000 wikiko_20210901_morpheme_mecab_fixed_decomposed_grammatical.txt wikiko_20210901_morpheme_mecab_fixed_decomposed_grammatical_




# LG
split -d -l 1500000 namuwiki_20200302_LG_mecab_fixed_composed_dummy_F.txt namuwiki_20200302_LG_mecab_fixed_composed_dummy_F_
split -d -l 1500000 wikiko_20210901_LG_mecab_fixed_composed_dummy_F.txt wikiko_20210901_LG_mecab_fixed_composed_dummy_F_

split -d -l 700000 namuwiki_20200302_LG_mecab_fixed_decomposed_pure_dummy_F.txt namuwiki_20200302_LG_mecab_fixed_decomposed_pure_dummy_F_
split -d -l 700000 wikiko_20210901_LG_mecab_fixed_decomposed_pure_dummy_F.txt wikiko_20210901_LG_mecab_fixed_decomposed_pure_dummy_F_

split -d -l 700000 namuwiki_20200302_LG_mecab_fixed_decomposed_lexical_dummy_F.txt namuwiki_20200302_LG_mecab_fixed_decomposed_lexical_dummy_F_
split -d -l 700000 wikiko_20210901_LG_mecab_fixed_decomposed_lexical_dummy_F.txt wikiko_20210901_LG_mecab_fixed_decomposed_lexical_dummy_F_


split -d -l 900000 namuwiki_20200302_LG_mecab_fixed_decomposed_grammatical_dummy_F.txt namuwiki_20200302_LG_mecab_fixed_decomposed_grammatical_dummy_F_
split -d -l 900000 wikiko_20210901_LG_mecab_fixed_decomposed_grammatical_dummy_F.txt wikiko_20210901_LG_mecab_fixed_decomposed_grammatical_dummy_F_












split -d -l 6000000 namuwiki_20200302_mecab_orig_composed.txt namuwiki_20200302_mecab_orig_composed_
split -d -l 3300000 namuwiki_20200302_mecab_orig_decomposed_pure.txt namuwiki_20200302_mecab_orig_decomposed_pure_
split -d -l 4000000 namuwiki_20200302_mecab_orig_decomposed_morphological.txt namuwiki_20200302_mecab_orig_decomposed_morphological_

split -d -l 6000000 namuwiki_20200302_tokenized_mecab_fixed_composed.txt namuwiki_20200302_tokenized_mecab_fixed_composed_
split -d -l 3000000 namuwiki_20200302_tokenized_mecab_fixed_decomposed_pure.txt namuwiki_20200302_tokenized_mecab_fixed_decomposed_pure_
split -d -l 5000000 namuwiki_20200302_tokenized_mecab_fixed_decomposed_morphological.txt namuwiki_20200302_tokenized_mecab_fixed_decomposed_morphological_

split -d -l 6000000 namuwiki_20200302_tokenized_mecab_fixed_composed.txt namuwiki_20200302_tokenized_mecab_fixed_composed_


split -d -l 1500000 wikiko_20210901_LG_mecab_fixed_composed_dummy_F.txt wikiko_20210901_LG_mecab_fixed_composed_dummy_F_
split -d -l 1500000 wikiko_20210901_LG_mecab_fixed_decomposed_grammatical_dummy_F.txt wikiko_20210901_LG_mecab_fixed_decomposed_grammatical_dummy_F_
split -d -l 800000 wikiko_20210901_LG_mecab_fixed_decomposed_lexical_dummy_F.txt wikiko_20210901_LG_mecab_fixed_decomposed_lexical_dummy_F_
split -d -l 800000 wikiko_20210901_LG_mecab_fixed_decomposed_pure_dummy_F.txt wikiko_20210901_LG_mecab_fixed_decomposed_pure_dummy_F_

```


namuwiki (without dummy letter) split
```bash
split -d -l 8000000 namuwiki_20200302_none_composed.txt namuwiki_20200302_none_composed_

split -d -l 6000000 namuwiki_20200302_mecab_orig_composed.txt namuwiki_20200302_mecab_orig_composed_
split -d -l 3300000 namuwiki_20200302_mecab_orig_decomposed_pure.txt namuwiki_20200302_mecab_orig_decomposed_pure_
split -d -l 4000000 namuwiki_20200302_mecab_orig_decomposed_morphological.txt namuwiki_20200302_mecab_orig_decomposed_morphological_

split -d -l 6000000 namuwiki_20200302_tokenized_mecab_fixed_composed.txt namuwiki_20200302_tokenized_mecab_fixed_composed_
split -d -l 3000000 namuwiki_20200302_tokenized_mecab_fixed_decomposed_pure.txt namuwiki_20200302_tokenized_mecab_fixed_decomposed_pure_
split -d -l 5000000 namuwiki_20200302_tokenized_mecab_fixed_decomposed_morphological.txt namuwiki_20200302_tokenized_mecab_fixed_decomposed_morphological_


split -d -l 6000000 namuwiki_20200302_tokenized_mecab_fixed_composed.txt namuwiki_20200302_tokenized_mecab_fixed_composed_


split -d -l 1500000 namuwiki_20200302_LG_mecab_fixed_composed_dummy_F.txt namuwiki_20200302_LG_mecab_fixed_composed_dummy_F_
split -d -l 2000000 namuwiki_20200302_LG_mecab_fixed_decomposed_grammatical_dummy_F.txt namuwiki_20200302_LG_mecab_fixed_decomposed_grammatical_dummy_F_
split -d -l 900000 namuwiki_20200302_LG_mecab_fixed_decomposed_lexical_dummy_F.txt namuwiki_20200302_LG_mecab_fixed_decomposed_lexical_dummy_F_
split -d -l 1000000 namuwiki_20200302_LG_mecab_fixed_decomposed_pure_dummy_F.txt namuwiki_20200302_LG_mecab_fixed_decomposed_pure_dummy_F_


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