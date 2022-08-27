from typing import List


from tokenizer.base import BaseTokenizer

from tokenizers import Tokenizer, decoders
from tokenizers import normalizers
from tokenizers.normalizers import StripAccents

from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing


class WordPieceTokenizer(BaseTokenizer):
    def __init__(self, model_path: str, reverse: bool = False, skip_special_tokens: bool = True):
        self.bert_tokenizer = Tokenizer.from_file(model_path)
        self.skip_special_tokens = skip_special_tokens
        # self.bert_tokenizer.decoder = decoders.WordPiece()

        # self.bert_tokenizer.normalizer = normalizers.Sequence([StripAccents()])  # normalizer
        # self.bert_tokenizer.pre_tokenizer = WhitespaceSplit()  # pretokenizer
        #
        # self.bert_tokenizer.post_processor = TemplateProcessing(
        #     single="[CLS] $A [SEP]",
        #     pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        #     special_tokens=[
        #         ("[CLS]", 1),
        #         ("[SEP]", 2),
        #     ],
        # )



    def tokenize(self, text: str) -> List[str]:
        output = self.bert_tokenizer.encode(text.strip())

        tokenized = self.bert_tokenizer.decode(output.ids, skip_special_tokens=False) # False로 하면 UNK 나옴

        tokenized = [token for token in tokenized.split(" ")]

        # print(output.tokens)
        # print(self.bert_tokenizer.decode(output.ids))

        return tokenized

    # def detokenize(self, tokens: List[str]) -> str:
    #     text = "".join(tokens).replace("▁", " ").strip()
    #     return text




# model_path = "./resources/v6_without_dummy_letter_grammatical_symbol_F/eojeol_mecab_fixed_composed_grammatical_symbol_F_wp-64k/tok.model"
# model_path = "./output_sp/eojeol_mecab_fixed_composed_wp-64k/tok.vocab"
# model_path = "./output_sp/eojeol_mecab_fixed_composed_wp-64k/tok.vocab"
#
# wp = WordPieceTokenizer(model_path=model_path)
#
# self = wp
#
# text = "내가 먹다"
# text = "나는 너를 사랑해"
# text = "전태희는 너를 사랑해"
# text = "나 ⫸는 너 ⫸ᆯ 좋아하 ⭧아"
# wp.tokenize(text)
#
#
# model_path="./output_sp/morpheme_mecab_orig_composed_sp-0k/tok.model"
# model_path="./output_sp/morpheme_mecab_orig_composed_sp-64k_0/tok.model"
#
# sp = spm.SentencePieceProcessor(model_file=model_path)
# sp.Encode("우리 ⭧는 좋 ⭧다", out_type=str)
#
# sp.encode("마을이 넓다", out_type)
# # model_path = "./resources/v3_without_dummy_letter/sp-32k/tok.model"
# sp = SentencePieceTokenizer(model_path)
# text = "대한민국에 우리끼리 살아보자"    # ['▁대한민국에', '▁우리', '끼리', '▁살아', '보자']
# text = "난 널 좋아해"  # ['▁난', '▁널', '▁좋아', '해']
# text = "밥을 먹습니다"  # ['▁밥', '을', '▁먹', '습니다']
#
# text = "우리 ⭧는 좋 ⭧다"
#
# text = '내 ⭧가 먹 ⭧다'
#
# ['내', '⭧가', '먹', '⭧다']
#
# sp.tokenize(text)
#
#
# self = sp