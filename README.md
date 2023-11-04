# Improving Korean NLP Tasks with Linguistically Informed Subword Tokenization and Sub-character Decomposition
- [arXiv link]

- Taehee Jeon*, Bongseok Yang*, Changxhwan Kim, Yoonseob Lim
  * *: Equal Contribution

- Abstract <br>
We introduce a morpheme-aware subword tokenization method that utilizes sub-character decomposition to address the challenges of applying Byte Pair Encoding (BPE) to Korean, a language characterized by its rich morphology and unique writing system. Our approach balances linguistic accuracy with computational efficiency in Pre-trained Language Models (PLMs). Our evaluations show that this technique achieves good performances overall, notably improving results in the syntactic task of NIKL-CoLA. This suggests that integrating morpheme type information can enhance language models' syntactic and semantic capabilities, indicating that adopting more linguistic insights can further improve performance beyond standard morphological analysis.



# Tutorial
- If you want to reproduce our experiment, please refer to [link](./Tutorial.md).



# Tokenization Methods
**Tokenization** | **Tokenized Sequence**
--- | ---
Raw Text | 나<u>라면</u> 해물<u>라면</u>을 먹었을걸.
<span style="display: block; border-bottom: 2px solid #000; margin: 10px 0;"></span> | <span style="display: block; border-bottom: 2px solid #000; margin: 10px 0;"></span>
WP | 나라/ \#\#면/ 해/ \#\#물/ \#\#라면/ \#\#을/ 먹었/ \#\#을/ \#\#걸 \#\#.
WP-SD | ㄴㅏㄹㅏ/ \#\#ㅁㅕㄴ/ ㅎㅐ/ \#\#ㅁㅜㄹ/ \#\#ㄹㅏㅁㅕㄴ/ \#\#ㅇㅡㄹ/ ㅁㅓㄱㅇㅓㅆㅇㅡㄹ/ \#\#걸.
MorWP | 나/ 이/ 라면/ 해물/ 라면/ 을/ 먹/ 었/ 을걸/ .
MorWP-SD | ㄴㅏ/ ㅇㅣ/ <i><u>ㄹㅏㅁㅕㄴ</u></i>/ ㅎㅐㅁㅜㄹ/  <u>ㄹㅏㅁㅕㄴ</u>/ ㅇㅡㄹ/ ㅁㅓㄱ/ ㅇㅓㅆ/ ㅇㅡㄹㄱㅓㄹ/ .
MorWP-MD | ㄴㅏ/ 이/ **<u>라면</u>**/ ㅎㅐㅁㅜㄹ/ **<u>ㄹㅏㅁㅕㄴ</u>**/ 을/  ㅁㅓㄱ/ 었/ 을걸/ .

1. WP (WordPiece): Subword Tokenization
2. WP-SD: Subword Tokenization with Simple Sub-character Decomposition
3. MorWP: Morpheme-aware Subword Tokenization
4. MorWP-SD: Morpheme-aware Subword Tokenization with Simple Sub-character Decomposition
5. MorWP-MD: Morpheme-aware Subword Tokenization with Morphological Sub-character Decomposition  



# Pre-training
We have trained BERT-Base models utilizing the official BERT codebase (https://github.com/google-research/bert). The pre-training corpus comprises dump files from Korean Wikipedia (https://dumps.wikimedia.org/kowiki/latest/) and Namuwiki (https://mu-star.net/wikidb), ensuring a comprehensive linguistic representation. Our training strictly follows empirical best practices, and we employ the following hyperparameters:

- Batch size: 1024
- Warm-up steps: 10,000
- Learning rate: 0.00005
- Maximum sequence length: 128
- Case sensitivity: Case-sensitive (do not convert to lowercase)
- Duplicate factor: 5
- Optimizer: AdamW

The training process is performed on a Google Cloud TPU v3-8 and typically completes within 4-5 days for each model.

**Acknowledgement**: For pre-training models, Cloud TPUs from the TensorFlow Research Cloud program were employed.


# Fine-tuning
- **Frameworks**: Pytorch, Huggingface Transformers
- **Tasks**: KLUE-NLI, NIKL-CoLA, NSMC, and others using adapted benchmark codes
  * KLUE-NLI, KLUE-DP: https://github.com/KLUE-benchmark/KLUE
  * PAWS-X: https://paperswithcode.com/dataset/paws-x
  * NIKL-CoLA (문법성 판단 말뭉치): https://corpus.korean.go.kr/request/reausetMain.do?lang=ko#down
  * NSMC: https://github.com/e9t/nsmc
  * HSD: https://github.com/kocohub/korean-hate-speech
- **Hyperparameters**:
- **Batch size**: 32 or 64
- **Learning rate**: 1e-5 to 5e-5
- **Epochs**: 1 to 10
- **Optimizer**: AdamW
- **Selection**: Chosen based on highest average dev set performance over 5 seeds; for KLUE-DP, best LAS and UAS averages
- **Hardware**: Each task fine-tuned on an individual RTX 2080 Ti GPU, with four used in total


## Results
| Vocab Size | Tokenization | NIKL-CoLA Dev | NIKL-CoLA Test | KLUE-DP UAS | KLUE-DP LAS | NSMC Dev | NSMC Test | HSD Dev | KLUE-NLI Dev | PAWS-X Dev | PAWS-X Test | OOV Rate (%) | Wordpiece Subtoken Rate (%) |
|------------|--------------|---------------|----------------|-------------|-------------|----------|-----------|---------|--------------|------------|-------------|--------------|-----------------------------|
| **32K**    | WP           | 57.62         | 61.64          | 92.56       | 87.02       | 90.06    | 89.52     | 64.82   | 76.20        | 76.60      | 72.39       | 0.78         | 54.63                       |
|            | WP-SD        | 59.61         | 59.85          | 92.58       | 87.21       | 89.69    | 89.38     | 64.09   | 76.23        | 78.20      | 75.23       | 0.57         | 51.42                       |
|            | MorWP        | 63.62         | 67.87          | 92.55       | 87.15       | 90.65    | 90.11     | 65.81   | 76.55        | 77.90      | 73.99       | 0.68         | 12.79                       |
|            | MorWP-SD     | 64.79         | 67.34          | 92.63       | 87.30       | 90.92    | 90.20     | 66.67   | 76.85        | 78.57      | 75.12       | 0.47         | 10.43                       |
|            | MorWP-MD     | 65.19         | 67.21          | 92.63       | ***87.30***       | 90.84    | 90.24     | 65.46   | 76.83        | 79.37      | 75.27       | 0.69         | 10.37                       |
| **64K**    | WP           | 59.15         | 60.21          | 92.65       | 87.06       | 89.73    | 89.53     | 61.98   | 76.99        | 78.14      | 73.57       | 0.90         | 47.96                       |
|            | WP-SD        | 58.76         | 60.91          | ***92.88***       | 87.12       | 89.82    | 89.63     | 62.20   | 76.72        | 79.33      | 74.49       | 0.63         | 46.58                       |
|            | MorWP        | 64.66         | 67.47          | 92.74       | 87.29       | 90.82    | ***90.40***     | 66.07   | 76.84        | ***79.76***      | 75.88       | 0.72         | 7.55                        |
|            | MorWP-SD     | 65.54         | 67.09          | 92.38       | 87.28       | ***90.96***    | 90.38     | ***68.55***   | 76.90        | 79.61      | 75.57       | 0.49         | 6.98                        |
|            | MorWP-MD     | ***66.32***      | ***69.64***          | 92.84       | 87.27       | 90.95    | 90.39     | 66.62   | ***78.01***        | 79.42      | ***76.22***       | 0.72         | 6.88                        |

Performance of various models on several NLP tasks and OOV rate, Wordpiece Subtoken Rate (WSR) of each model. The best scores in each column are bold-faced and italicized. The metrics of each task are as follows:
- NIKL-CoLA: Accuracy
- KLUE-DP: Macro F1 (UAS, LAS)
- NSMC: Accuracy
- HSD: Macro F1
- KLUE-NLI: Accuracy
- PAWS-X: Accuracy.

OOV Rate, WSR are caculated as:
- OOV Rate = The number of OOV tokens / The number of tokens * 100
- WSR = The number of Wordpiece subtokens / The number of tokens * 100


# Citation
citation

