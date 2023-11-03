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
  * Wikiextractor를 이용한 텍스트 추출 과정은 본 저장소의 코드에 포함되어 있지 않음. 자세한 이용 방법은 https://github.com/attardi/wikiextractor 참고할 것. 

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
- 다운받은 덤프 파일의 압축을 푼 후, json 파일 (예: namuwiki200302.json) 다음 경로에 위치시키기.
  * ./corpus/raw_corpus


# 1. 코퍼스 전처리하기
```bash
python ./scripts/parse_Wikiko_with_preprocessing.py --input_path ./corpus/raw_corpus/text --output_path ./corpus/preprocessed/wikiko_20210901_preprocessed.txt
python ./scripts/parse_namuWiki_with_preprocessing.py --input_path ./corpus/raw_corpus/namuwiki200302.json --output_path ./corpus/preprocessed/namuwiki_20200302_preprocessed.txt   
```
- 출력 파일은 ./pretrain_corpus/preprocessed 에 저장됨.


# 2. 코퍼스 토큰화
- *token_type*과 *decomposition_type*은 paper의 tokenization methods와 관련됨.
  * WP: --token_type=eojeol --decomposition_type=composed
  * WP-SD --token_type=eojeol --decomposition_type=decomposed_simple
  * MorWP: --token_type=morpheme --decomposition_type=composed
  * MorWP-SD: --token_type=morpheme --decomposition_type=decomposed_simple
  * MorWP-MD: --token_type=morpheme --decomposition_type=decomposed_lexical

## wiki-ko
```bash
python scripts/mecab_tokenization.py --token_type=eojeol --decomposition_type=composed --corpus_path=./corpus/preprocessed/wikiko_20210901_preprocessed.txt --tokenizer_type=mecab_fixed --threads 32
python scripts/mecab_tokenization.py --token_type=eojeol --decomposition_type=decomposed_simple --corpus_path=./corpus/preprocessed/wikiko_20210901_preprocessed.txt --tokenizer_type=mecab_fixed --nfd --threads 32

python scripts/mecab_tokenization.py --token_type=morpheme --decomposition_type=composed --corpus_path=./corpus/preprocessed/wikiko_20210901_preprocessed.txt --tokenizer_type=mecab_fixed --threads 32
python scripts/mecab_tokenization.py --token_type=morpheme --decomposition_type=decomposed_simple --corpus_path=./corpus/preprocessed/wikiko_20210901_preprocessed.txt --tokenizer_type=mecab_fixed --nfd --threads 32
python scripts/mecab_tokenization.py --token_type=morpheme --decomposition_type=decomposed_lexical --corpus_path=./corpus/preprocessed/wikiko_20210901_preprocessed.txt --tokenizer_type=mecab_fixed --nfd --threads 32 
```

## namuwiki
```bash
python scripts/mecab_tokenization.py --token_type=eojeol --corpus_path=./corpus/preprocessed/namuwiki_20200302_preprocessed.txt --tokenizer_type=mecab_fixed --decomposition_type=composed --threads 32
python scripts/mecab_tokenization.py --token_type=eojeol --corpus_path=./corpus/preprocessed/namuwiki_20200302_preprocessed.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_simple --nfd --threads 32

python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_preprocessed.txt --tokenizer_type=mecab_fixed --decomposition_type=composed --threads 32
python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_preprocessed.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_simple --nfd --threads 32 
python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_preprocessed.txt --tokenizer_type=mecab_fixed --decomposition_type=decomposed_lexical --nfd --threads 32
```

- 출력 파일은 ./pretrain_corpus/tokenized 에 저장됨.


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



# 3. Wordpiece 학습
- 32k
```bash
python scripts/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/eojeol_mecab_fixed/composed_dummy_F --vocab_size=32000
python scripts/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/eojeol_mecab_fixed/decomposed_simple_dummy_F --vocab_size=32000

python scripts/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized//morpheme_mecab_fixed/composed_dummy_F --vocab_size=32000
python scripts/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized//morpheme_mecab_fixed/decomposed_simple_dummy_F --vocab_size=32000
python scripts/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized//morpheme_mecab_fixed/decomposed_lexical_dummy_F --vocab_size=32000
````

- 64k
```bash
python scripts/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/eojeol_mecab_fixed/composed_dummy_F --vocab_size=64000
python scripts/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/eojeol_mecab_fixed/decomposed_simple_dummy_F --vocab_size=64000

python scripts/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized//morpheme_mecab_fixed/composed_dummy_F --vocab_size=64000
python scripts/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized//morpheme_mecab_fixed/decomposed_simple_dummy_F --vocab_size=64000
python scripts/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized//morpheme_mecab_fixed/decomposed_lexical_dummy_F --vocab_size=64000
````

- 출력 파일은 ./resources에 저장됨.

[//]: # ([comment]: <> &#40;## mecab 토큰화&#41;)

[//]: # ()
[//]: # ([comment]: <> &#40;./build_vocab/build_mecab_vocab_our.py &#40;자동화 위해 코드 수정 필요&#41;&#41;)

[//]: # ()
[//]: # ([comment]: <> &#40;```bash&#41;)

[//]: # ()
[//]: # ([comment]: <> &#40;python build_vocab/build_mecab_vocab_our.py --vocab_size=64000&#41;)

[//]: # ()
[//]: # ([comment]: <> &#40;```&#41;)

[//]: # ()
[//]: # ()
[//]: # (- 현재는 ./output_sp 디렉토리에 출력됨. 추후 resources로 가도록 수정해야 함.)

[//]: # ()
[//]: # ()
[//]: # (## Wordpiece)

[//]: # (./build_vocab/train_wordsentencepiece.py)

[//]: # (### *baselines*)

[//]: # (```bash)

[//]: # (python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/eojeol_mecab_fixed/composed_dummy_F --vocab_size=64000)

[//]: # (python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/eojeol_mecab_fixed/decomposed_pure_dummy_F --vocab_size=64000)

[//]: # ()
[//]: # (python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized//morpheme_mecab_fixed/composed_dummy_F --vocab_size=64000)

[//]: # (python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized//morpheme_mecab_fixed/decomposed_lexical_dummy_F --vocab_size=64000)

[//]: # (python build_vocab/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized//morpheme_mecab_fixed/decomposed_pure_dummy_F --vocab_size=64000)

[//]: # ()
[//]: # (```)


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



# 4. make BERT files 
- BERT 학습에 필요한 파일 생성하기.
```bash
python scripts/make_bert_files.py --root_path=resources --model_max_length=128
```
실행 후 파일들 resources로 옮기기. 수동으로.  # 추후 자동화.



# 5. pretrain BERT
- Make tfrecord files for BERT pretraining.
  * We used the official code (https://github.com/google-research/bert).
-  It is highly recommended to split corpus files into multiple files for better time-saving.  

## 1) 토큰화된 코퍼스 분할
- ./corpus/tokinized 에 위치한 토큰화된 코퍼스 파일들을 분할
- example 1: WP
  ```bash
  split -d -l wikiko_20210901_eojeol_mecab_fixed_composed_dummy_F.txt wikiko_20210901_eojeol_mecab_fixed_composed_dummy_F_
  split -d -l namuwiki_20200302_eojeol_mecab_fixed_composed_dummy_F.txt namuwiki_20200302_eojeol_mecab_fixed_composed_dummy_F_
  ```
- example 2: MorWP
  ```bash
  split -d -l wikiko_20210901_morpheme_mecab_fixed_decomposed_lexical_dummy_F.txt wikiko_20210901_morpheme_mecab_fixed_decomposed_lexical_dummy_F_
  split -d -l namuwiki_20200302_morpheme_mecab_fixed_decomposed_lexical_dummy_F.txt namuwiki_20200302_morpheme_mecab_fixed_decomposed_lexical_dummy_F_
  ```

## 2) pretraining
- Follow the instructions in the official github repository of BERT (https://github.com/google-research/bert).

## input
- tfrecord files: from tokenized corpus files
- tok.vocab: in ./resources/**tokenization method & vocab size**/
- bert_config.json: in ./resources/**tokenization method & vocab size**/

- *Please make sure that the input files (tfrecord fiels, tok.vocab, bert_config.json) have the same tokenized method and vocab size when you pretrain a BERT model.*