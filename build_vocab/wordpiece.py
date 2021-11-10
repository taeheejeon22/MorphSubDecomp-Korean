from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer(
    vocab_file=None,
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=False, # Must be False if cased model
    lowercase=False,
    wordpieces_prefix="##"
)


tokenizer.train(
    files=["./corpus/tokenized/without_dummy_letter/wikiko_20210901_mecab_fixed/decomposed_morphological/wikiko_20210901_mecab_fixed_decomposed_morphological.txt"],
    limit_alphabet=6000,
    vocab_size=32000,
)

tokenizer.save("./wp_sample")


# vocab 전처리
import json # import json module

vocab_path = "./wp_sample"

vocab_file = "./wp_sample.txt"
f = open(vocab_file,'w',encoding='utf-8')
with open(vocab_path) as json_file:
    json_data = json.load(json_file)
    for item in json_data["model"]["vocab"].keys():
        f.write(item+'\n')

    f.close()


# test
from transformers.tokenization_bert import BertTokenizer

vocab_path = "./wp_sample.txt"
tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=False)

text = ""


text = '[CLS] 나 너 [SEP]'
text = '당신을 좋아해'
text = '주니어 ㄴㅡㄴ 대통령 이 ㄷㅏ .'


text = '너를 좋아하는데'
text = '네가 예쁜데'

tokenizer.decode( tokenizer.encode(text, add_special_tokens=False) )
