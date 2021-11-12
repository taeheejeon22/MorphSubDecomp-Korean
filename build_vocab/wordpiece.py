from tokenizers import Tokenizer
from tokenizers.models import WordPiece

bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))


## normalizers
# from tokenizers import normalizers
# from tokenizers.normalizers import Lowercase, NFD, StripAccents
#
# bert_tokenizer.normalizer = normalizers.Sequence([StripAccents()])


## pre-tokenizer
from tokenizers.pre_tokenizers import Whitespace, Metaspace, WhitespaceSplit
# bert_tokenizer.pre_tokenizer = Whitespace()
# bert_tokenizer.pre_tokenizer = Metaspace()
bert_tokenizer.pre_tokenizer = WhitespaceSplit()



from tokenizers.processors import TemplateProcessing

bert_tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 1),
        ("[SEP]", 2),
    ],
)


## train
from tokenizers.trainers import WordPieceTrainer
trainer = WordPieceTrainer(
    vocab_size=16000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
)

files = ["./corpus/tokenized/space_F_dummy_F_grammatical_T_0/wikiko_20210901_morpheme_mecab_orig/composed/wikiko_20210901_morpheme_mecab_orig_composed.txt"]
bert_tokenizer.train(files, trainer)


bert_tokenizer.save("./bert-wiki.json")


import re
p = re.compile("\w+|[^\w\s]+")
p.search("⭧은")


out = bert_tokenizer.encode("은 ⭧은 좋 ⭧다")
print(out.ids)
print(out.tokens)

bert_tokenizer.decode([1, 2309, 98, 2309, 2459, 98, 819, 2])


from tokenizers import decoders

bert_tokenizer.decoder = decoders.WordPiece()
bert_tokenizer.decode(out.ids)


#####################################################################################################################
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
    files=["./corpus/tokenized/space_F_dummy_F_grammatical_T_0/wikiko_20210901_morpheme_mecab_orig/decomposed_lexical/wikiko_20210901_morpheme_mecab_orig_decomposed_lexical.txt"],
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
text = "은 ⭧은 좋 ⭧다"

text = '너를 좋아하는데'
text = '네가 예쁜데'

tokenizer.encode(text, add_special_tokens=False)
tokenizer.decode( tokenizer.encode(text, add_special_tokens=False) )
