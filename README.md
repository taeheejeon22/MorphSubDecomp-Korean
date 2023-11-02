# Tutorial

![mecab_fixed](https://github.com/taeheejeon22/konlpy-mecab-fixed)
<br>
![kortok](https://github.com/kakaobrain/kortok)


# tokenization stratagies
## eojeol
- composed                # WP only
- decomposed simple (nfd)   # KR-BERT

### morpheme_fixed
- composed # kortok
- decomposed simple
- decomposed lexical


# 0. raw 코퍼스 준비하기
## 1) wiki-ko
### 1-1) 파일 받기
- 덤프 파일 저장소: https://dumps.wikimedia.org/kowiki/latest/
  * pages-articles.xml.bz2 파일 받기.
  * 실험에 사용한 파일의 버전: 21/09/01

### 1-2) 텍스트 추출
- 다운받은 덤프 파일로부터 Wikiextractor (https://github.com/attardi/wikiextractor) 이용하여 텍스트 추출함. 

### 1-3) 추출된 텍스트 파일 옮기기
- Wikiextractor 사용 후 생성된 'text' 폴더를 다음 경로에 위치시키기. 
  * ./corpus/raw_corpus

## 2) namuwiki
### 2-1) 파일 받기
- 덤프 파일 저장소: https://mu-star.net/wikidb
  * 실험에 사용한 파일은 http://dump.thewiki.kr/ 에서 받은 것이나, 2023년 11월 현재 접속 불가한 듯함. 위 사이트에서도 동일한 파일을 제공하는 것으로 보임.
  * 실험에 사용한 파일의 버전: 20/03/02

### 2-2) 텍스트 추출
- namuwiki 덤프 파일의 텍스트 추출 과정은 전처리 과정과 통합되어 있으므로 생략함.

### 2-3) 덤프 파일 옮기기
- 다운받은 덤프 파일의 압축을 푼 후, json 파일 (예: namuwiki200302.json)을, .



# 1. 코퍼스 전처리하기
```bash
python ./scripts/parse_Wikiko_with_preprocessing.py --input_path ./corpus/raw_corpus/text --output_path ./corpus/preprocessed/wikiko_20210901_with_preprocessing.txt
python ./scripts/parse_namuWiki_with_preprocessing.py --input_path ./corpus/raw_corpus/namuwiki200302.json --output_path ./corpus/preprocessed/namuwiki_20200302_with_preprocessing.txt   
```



# 2. 코퍼스 토큰화
- ./scripts/mecab_tokenization_v2_1.py
- 출력 파일은 ./pretrain_corpus/tokenized 에 저장됨.

## wiki-ko
```bash
python scripts/mecab_tokenization.py --token_type=eojeol --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing.txt --tokenizer_type=mecab_fixed --decomposition_type=composed --threads 32
python scripts/mecab_tokenization.py --token_type=eojeol --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_simple --nfd --threads 32

python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing.txt --tokenizer_type=mecab_fixed --decomposition_type=composed --threads 32
python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_simple --nfd --threads 32
python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/wikiko_20210901_with_preprocessing.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_lexical --nfd --threads 32 
```

## namuwiki
```bash
python scripts/mecab_tokenization.py --token_type=eojeol --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing.txt --tokenizer_type=mecab_fixed --decomposition_type=composed --threads 32
python scripts/mecab_tokenization.py --token_type=eojeol --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_simple --nfd --threads 32

python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing.txt --tokenizer_type=mecab_fixed --decomposition_type=composed --threads 32
python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_simple --nfd --threads 32 
python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_lexical --nfd --threads 32
```

### 메모리 부족으로 인해 코드 중단되는 경우의 대안
1. 전처리된 코퍼스 파일 분할
```bash
split -d -l 11000000 namuwiki_20200302_with_preprocessing.txt namuwiki_20200302_with_preprocessing_
```

2. 분할한 파일 각각에 대해 토큰화 수행
- decomposed_simple
```bash
python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_00 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_simple --nfd --threads 32
python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_01 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_simple --nfd --threads 32
python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_02 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_simple --nfd --threads 32
```

- decomposed_lexical
```bash
python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_00 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_lexical --nfd --threads 32
python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_01 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_lexical --nfd --threads 32
python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_02 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_lexical --nfd --threads 32
```

- 다른 경우(예: wiki-ko의 decomposed_pure, namuwiki의 composed)에 대해서도 메모리 부족 문제가 발생한다면, 위와 같은 방식 활용하면 됨.




# 2. Wordpiece, Sentencepiece 학습
[comment]: <> (## mecab 토큰화)

[comment]: <> (./build_vocab/build_mecab_vocab_our.py &#40;자동화 위해 코드 수정 필요&#41;)

[comment]: <> (```bash)

[comment]: <> (python build_vocab/build_mecab_vocab_our.py --vocab_size=64000)

[comment]: <> (```)


- 현재는 ./output_sp 디렉토리에 출력됨. 추후 resources로 가도록 수정해야 함.


## Wordpiece
./build_vocab/train_wordsentencepiece.py
### *baselines*
```bash
python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/eojeol_mecab_fixed/composed_dummy_F --vocab_size=64000
python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/eojeol_mecab_fixed/decomposed_pure_dummy_F --vocab_size=64000

python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized//morpheme_mecab_fixed/composed_dummy_F --vocab_size=64000
python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized//morpheme_mecab_fixed/decomposed_lexical_dummy_F --vocab_size=64000
python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized//morpheme_mecab_fixed/decomposed_pure_dummy_F --vocab_size=64000

```


[//]: # (## Sentencepiece)

[//]: # (./build_vocab/train_sentencepiece.py)

[//]: # (### *baselines*)

[//]: # (```bash)

[//]: # (python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=eojeol --tokenizer_type=mecab_fixed --composition_type=composed)

[//]: # (python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=eojeol --tokenizer_type=mecab_fixed --composition_type=decomposed_pure)

[//]: # ()
[//]: # (python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_orig --composition_type=composed)

[//]: # (python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_orig --composition_type=decomposed_pure)

[//]: # (python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_fixed --composition_type=composed)

[//]: # (python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_F --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_fixed --composition_type=decomposed_pure)

[//]: # (```)

[//]: # ()
[//]: # (### *OUR*)

[//]: # (```bash)

[//]: # (python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_T --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_orig --composition_type=composed)

[//]: # (python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_T --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_orig --composition_type=decomposed_lexical)

[//]: # (python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_T --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_orig --composition_type=decomposed_pure)

[//]: # ()
[//]: # (python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_T --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_fixed --composition_type=composed)

[//]: # (python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_T --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_fixed --composition_type=decomposed_lexical)

[//]: # (python build_vocab/train_sentencepiece.py --tokenized_corpus_path=./corpus/tokenized/space_F_dummy_F_grammatical_T --vocab_size=64000 --token_type=morpheme --tokenizer_type=mecab_fixed --composition_type=decomposed_pure)

[//]: # (```)






# 3. make BERT files 
```bash
python scripts/make_bert_files.py --root_path=output_sp/ --vocab_size=64000 --model_max_length=128
```
실행 후 파일들 resources로 옮기기. 수동으로.  # 추후 자동화.



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
