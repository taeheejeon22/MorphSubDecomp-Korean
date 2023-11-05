# Tutorial

# tokenization stratagies
- WP (WordPiece): Subword Tokenization
- WP-SD: Subword Tokenization with Simple Sub-character Decomposition
- MorWP: Morpheme-aware Subword Tokenization
- MorWP-SD: Morpheme-aware Subword Tokenization with Simple Sub-character Decomposition
- MorWP-MD: Morpheme-aware Subword Tokenization with Morphological Sub-character Decomposition  

# 0. Preparing the Raw Corpus
## 1) wiki-ko
### 1-1) Dowloading the File
- Dump file repository: https://dumps.wikimedia.org/kowiki/latest/
  * Download the *pages-articles.xml.bz2* file.
  * Version of the file used in the experiment: 09/01/21 (MM/DD/YY)

### 1-2) Extracting Text
- Use Wikiextractor (https://github.com/attardi/wikiextractor) to extract text from the downloaded dump file.
  * The text extraction process using Wikiextractor is not included in the code of this repository. For detailed usage, refer to https://github.com/attardi/wikiextractor. 

### 1-3) Moving the Extracted Text Files
- After using Wikiextractor, place the created 'text' folder in the following path: 
  * ./corpus/raw_corpus

## 2) namuwiki
### 2-1) Dowloading the File
- Dump file repository: https://mu-star.net/wikidb
  * The file used in the experiment was downloaded from http://dump.thewiki.kr/, but it seems to be inaccessible as of November 2023. The above site appears to provide the same file.
  * Version of the file used in the experiment: 03/02/20 (MM/DD/YY)

### 2-2) Extracting Text
- The text extraction process for namuwiki dump files is integrated with the preprocessing steps, so it is omitted here.

### 2-3) Moving the Dump File
- After decompressing the downloaded dump file, place the json file (e.g., namuwiki200302.json) in the following path:
  * ./corpus/raw_corpus



# 1. Preprocessing the Corpus
```bash
python ./scripts/parse_Wikiko_with_preprocessing.py --input_path ./corpus/raw_corpus/text --output_path ./corpus/preprocessed/wikiko_20210901_preprocessed.txt
python ./scripts/parse_namuWiki_with_preprocessing.py --input_path ./corpus/raw_corpus/namuwiki200302.json --output_path ./corpus/preprocessed/namuwiki_20200302_preprocessed.txt   
```
- The output files are saved in the following path:
  * ./pretrain_corpus/preprocessed



# 2. Tokenization of Corpus
- *token_type* and *decomposition_type* relate to the tokenization methods discussed in the paper.
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

- The output files are stored in the following path:
  * ./pretrain_corpus/tokenized


### Alternative solutions in case the code is interrupted due to insufficient memory
1. Split the preprocessed corpus file
```bash
split -d -l 11000000 namuwiki_20200302_with_preprocessing.txt namuwiki_20200302_with_preprocessing_
```

2. Perform tokenization on each split file
- For the *decomposed_simple* method:
```bash
python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_00 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_simple --nfd --threads 32
python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_01 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_simple --nfd --threads 32
python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_02 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_simple --nfd --threads 32
```

- For the *decomposed_lexical* method:
```bash
python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_00 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_lexical --nfd --threads 32
python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_01 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_lexical --nfd --threads 32
python scripts/mecab_tokenization.py --token_type=morpheme --corpus_path=./corpus/preprocessed/namuwiki_20200302_with_preprocessing_02 --tokenizer_type=mecab_fixed --decomposition_type=decomposed_lexical --nfd --threads 32
```

- If memory issues occur with other methods (e.g., *decomposed_pure* for wiki-ko, composed for namuwiki), the same splitting strategy can be employed.



# 3. Wordpiece Model Training
For training WordPiece models with different vocabulary sizes, the following bash commands can be used. The first set is for a vocabulary size of 32,000, and the second set is for 64,000. The output files from this training will be stored in the ./resources directory.

- For a vocabulary size of 32k:
```bash
python scripts/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/eojeol_mecab_fixed/composed_dummy_F --vocab_size=32000
python scripts/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/eojeol_mecab_fixed/decomposed_simple_dummy_F --vocab_size=32000

python scripts/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized//morpheme_mecab_fixed/composed_dummy_F --vocab_size=32000
python scripts/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized//morpheme_mecab_fixed/decomposed_simple_dummy_F --vocab_size=32000
python scripts/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized//morpheme_mecab_fixed/decomposed_lexical_dummy_F --vocab_size=32000
````

- For a vocabulary size of 64k:
```bash
python scripts/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/eojeol_mecab_fixed/composed_dummy_F --vocab_size=64000
python scripts/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized/eojeol_mecab_fixed/decomposed_simple_dummy_F --vocab_size=64000

python scripts/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized//morpheme_mecab_fixed/composed_dummy_F --vocab_size=64000
python scripts/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized//morpheme_mecab_fixed/decomposed_simple_dummy_F --vocab_size=64000
python scripts/train_wordpiece.py --tokenized_corpus_path=./corpus/tokenized//morpheme_mecab_fixed/decomposed_lexical_dummy_F --vocab_size=64000
````

- The output files are stored in the following path:
  * ./resources



# 4. Creating Files for BERT Training 
To generate the necessary files for training a BERT model, you can use the following bash command. This script will take the trained WordPiece models and other resources to prepare the input files that BERT needs for pretraining:
```bash
python scripts/make_bert_files.py --root_path=resources --model_max_length=128
```
- This command assumes that all the necessary resources, including the WordPiece vocabulary files and tokenized corpus, are stored under the ***resources*** directory. The --model_max_length argument specifies the maximum sequence length that the model will support. In this case, sequences will be padded or truncated to a length of 128 tokens.



# 5. Pre-training BERT
- To pretrain a BERT model, one needs to first convert the tokenized corpus into a format that BERT understands - typically tfrecord files. As you pointed out, splitting the corpus into multiple smaller files can be beneficial for managing resources and improving training time efficiency.


## 1) Splitting the Tokenized Corpus
You have provided commands for splitting the tokenized corpus into smaller files, which is done to facilitate the handling during the creation of tfrecord files for BERT pretraining.

- For WordPiece (WP):
  ```bash
  split -d -l 1000000 wikiko_20210901_eojeol_mecab_fixed_composed_dummy_F.txt wikiko_20210901_eojeol_mecab_fixed_composed_dummy_F_
  split -d -l 1000000 namuwiki_20200302_eojeol_mecab_fixed_composed_dummy_F.txt namuwiki_20200302_eojeol_mecab_fixed_composed_dummy_F_
  ```

- For Morpheme WordPiece (MorWP):
  ```bash
  split -d -l 1000000 wikiko_20210901_morpheme_mecab_fixed_decomposed_lexical_dummy_F.txt wikiko_20210901_morpheme_mecab_fixed_decomposed_lexical_dummy_F_
  split -d -l 1000000 namuwiki_20200302_morpheme_mecab_fixed_decomposed_lexical_dummy_F.txt namuwiki_20200302_morpheme_mecab_fixed_decomposed_lexical_dummy_F_
  ```
- Please note that the -l option in the split command dictates the number of lines each split file should contain. Adjust this number based on the size of your corpus and the memory limitations of your training environment.


## 2) Pre-training
Following the official BERT GitHub repository (https://github.com/google-research/bert)'s instructions is crucial. Here is a simplified overview of the steps:

1. **Create tfrecord files**: Using the BERT repository scripts, convert the split tokenized corpus files into tfrecord files.
2. **Prepare configuration files**: Make sure you have the correct bert_config.json configuration file which matches the architecture of the model you're planning to pre-trian.
3. **Set up training environment**: Make sure your training environment is correctly configured with all necessary libraries and dependencies installed.
4. **Begin pre-trianing**: Using the BERT repository's pre-trianing script, start the training process with the appropriate flags set for your tfrecord files, vocabulary file, and configuration file.

## Input Files for the Pre-training Process
- **tfrecord files**: Generated from the split tokenized corpus files.
- **tok.vocab**: Located in ./resources/**tokenization method & vocab size**/
- **bert_config.json**: Located in ./resources/**tokenization method & vocab size**/

*It's essential that all input files (tfrecord files, tok.vocab, bert_config.json) are consistent in terms of the tokenization method and vocabulary size used. Inconsistencies could lead to errors or suboptimal training results.*



# 6. Finetuning
- We will now use the pre-trained BERT to perform downstream tasks. The excution methods of KLUE-NLI, KLUE-DP, and NIKL-CoLA, PAWS, NSMC, and HSD are different, respectively.

## Dataset
- Information on each dataset is presented in the following table.

|Dataset|Link|Paper|
|---|---|---|
|KLUE-DP, KLUE-NLI|https://github.com/KLUE-benchmark/KLUE/tree/1cc52e64c0e0b6915577244f7439c55a42199a64|[Park et al. (2021)](https://arxiv.org/abs/2105.09680)
|HSD|https://github.com/kocohub/korean-hate-speech |[Moon et al. (2020)](https://aclanthology.org/2020.socialnlp-1.4/) |
|NSMC|https://github.com/e9t/nsmc| - |
|NIKL-CoLA|https://corpus.korean.go.kr/request/reausetMain.do?lang=ko | - |
|PAWS-X| https://github.com/google-research-datasets/paws/tree/master/pawsx | [Yang et al. (2019)](https://arxiv.org/abs/1908.11828)|

## 1) Convert TensorFlow model to PyTorch model  
- Our pre-trained checkpoint is the TensorFlow-based model, and the fine-tuning framework is the PyTorch and Transformers. Therefore, it is necessary to convert the tenserflow model so that it can be used in pytorch and transformers framework. 
- The output file should be located in `./resources/{tokenizer_name}`. 
- For example, iIf you have a TensorFlow checkpoint file created using the MorWP-MD-64k tokenizer, You need to run below:
```
transformers-cli convert --model_type bert\
  --tf_checkpoint=./bert_model_morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64000.ckpt-1000000 \
  --config=./resources/morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64000/config.json \
  --pytorch_dump_output=./resources/morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64000/pytorch_model.bin
```

- *Note*: The name of the output model file should be ***'pytorch_model.bin'***.



## 2) KLUE-tasks
Fine-tuning of KLUE-tasks is performed according to the [KLUE_baseline repository](https://github.com/KLUE-benchmark/KLUE-baseline).

- Run the file `run_klue.py` as follows.
- The arguments `model_name_or_path` and `tokenizer_name` and `config_name` should be directory paths. The directory path for both are set to combination of `./resources` and the name of tokenzier.
- For example, if you run fine-tuning with MorWP-64k model, run code below:
- The hyperparameters listed below are the ones that produced the best scores in our research.

1. KLUE-DP
```bash
python run_klue.py train \
--task klue-dp \
--output_dir ./run_outputs \
--data_dir ./KLUE-baseline/data/klue_benchmark/klue-dp-1.1 \
--model_name_or_path ./resources/morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64000 \
--tokenizer_name ./resources/morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64000 \
--config_name ./resources/morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64000 \
--learning_rate 5e-5
--train_batch_size 32 
--num_train_epochs 10 \
--metric_key uas_macro_f1 --gpus 0 --num_workers 8 \
--seed 42
```

2. KLUE-NLI
```bash
python run_klue.py train \
--task klue-nli \
--output_dir ./run_outputs \
--data_dir ./KLUE-baseline/data/klue_benchmark/klue-nli-1.1 \
--model_name_or_path ./resources/morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64000 \
--tokenizer_name ./resources/morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64000\
--config_name ./resources/morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64000 \
--learning_rate 3e-5
--train_batch_size 32 
--num_train_epochs 5 
--metric_key accuracy --gpus 0 --num_workers 8 \
--seed 42
```

## 3) HSD, NSMC, NIKL-CoLA, PAWS-X
- Run the file `tasks/{task}/run_train.py`.
- The argument `resource_dir` should be directory path. The directory path for `resource_dir` is set to combination of `./resources` and each tokenizer name.
- The argument `tokenizer` should be tokenizer name.
- For example, if you run fine-tuning with MorWP-64k model, run code below:
- The hyperparameters listed below are the ones that produced the best scores in our research.

1. HSD
```bash
python ./tasks/hsd/run_train.py \
--tokenizer morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64000 \
--resource_dir ./resources/morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64000 \
--batch_size 32 \
--learning_rate 5e-4 \
--num_epochs 4 \
--seed 42
--summary_dir ./run_outputs/summary
```

1. NSMC
```bash
python ./tasks/nsmc/run_train.py \
--tokenizer morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64000 \
--resource_dir ./resources/morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64000 \
--batch_size 64 \
--learning_rate 2e-5 \
--num_epochs 2 \
--seed 42
--summary_dir ./run_outputs/summary
```

1. NIKL-CoLA
```bash
python ./tasks/cola/run_train.py \
--tokenizer morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64000 \
--resource_dir ./resources/morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64000 \
--batch_size 64 \
--learning_rate 1e-5 \
--num_epochs 3 \
--seed 42
--summary_dir ./run_outputs/summary
```

1. PAWS-X
```bash
python ./tasks/paws/run_train.py \
--tokenizer morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64000 \
--resource_dir ./resources/morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64000 \
--batch_size 64 \
--learning_rate 5e-5 \
--num_epochs 5 \
--seed 42
--summary_dir ./run_outputs/summary
```
