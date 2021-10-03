# ours
import json
import os
from typing import List

import MeCab

from itertools import chain
from tokenizer.base import BaseTokenizer

from konlpy.tag import Mecab

import scripts.tokenizers_acl as tok





class MeCabTokenizer(BaseTokenizer):
    # def __init__(self, use_original: bool, pure_decompostion: bool, morphological: bool):
    def __init__(self, config_path: str):
        # self.mecab = MeCab.Tagger(f"--dicdir /usr/local/lib/mecab/dic/mecab-ko-dic")
        with open(config_path) as f:
            self.config: dict = json.load(f)  ### space_symbol이 뭔지만 알면 쓸모 없는 듯...


        # # without config file
        # self.tok = tok.tokenizers(dummy_letter="⊸", space_symbol = "▃")
        # self.use_original = use_original
        # self.pure_decomposition = pure_decompostion
        # self.morphological = morphological

        # with config file
        self.tok = tok.tokenizers(dummy_letter=self.config["dummy_letter"], space_symbol=self.config["space_symbol"])
        self.use_original = self.config["use_original"]
        self.pure_decomposition = self.config["pure_decomposition"]
        self.morphological = self.config["morphological"]


    # # for inserting space_symbol ("▃")
    # # https://stackoverflow.com/questions/5920643/add-an-item-between-each-item-already-in-the-list
    # def intersperse(self, lst, item):
    #     result = [item] * (len(lst) * 2 - 1)
    #     result[0::2] = lst
    #     return result
    #
    #
    # def tokenize(self, text):  # "▃"
    #     mor_poss = self.mc.pos(text, flatten=False)  # [[('이것', 'NP'), ('이', 'JKC')], [('아니', 'VCN'), ('다', 'EC')]]
    #     mors = [[mor_pos[0] for mor_pos in word] for word in mor_poss]  # [['이것', '이'], ['아니', '다']]
    #     # return list(chain.from_iterable(self.intersperse(mors, self.config["space_symbol"])))  # ['이것', '이', '▃', '아니', '다']
    #
    #     return list(chain.from_iterable(self.intersperse(mors, "▃")))  # ['이것', '이', '▃', '아니', '다']


    # def detokenize(self, tokens: List[str]) -> str:
    #     text = "".join(tokens).replace("▃", " ").strip()
    #     return text


    def tokenize(self, text: str):
        return self.tok.mecab_tokenizer(text, use_original=self.use_original, pure_decomposition=self.pure_decomposition, morphological=self.morphological)


#     def detokenize(self, tokens: List[str]) -> str:
#         joined = " ".join( [word.replace(" " , "") for word in " ".join(tokens).split(" ▃ ")] ) # 'ㅅㅏ⊸ㄹㅏㅁㅇㅡㄴ ㄴㅓ⊸ㄹㅡㄹ ㅇㅝㄴㅎㅐ⊸.'
#
#         if self.pure_decomposition == True:
#             detokenized = self.tok.jamo2str(joined)
#         elif self.pure_decomposition == False:
#             # detokenized = self.tok.jamo2str(joined)
#             detokenized = self.tok.jamo2str_morphological(joined)
#
#         return detokenized
#
#
#
#
# mc = MeCabTokenizer(use_original=True, pure_decompostion=True, morphological=False)
# mc = MeCabTokenizer(use_original=True, pure_decompostion=False)
# mc = MeCabTokenizer(use_original=False, pure_decompostion=True)
# mc = MeCabTokenizer(use_original=False, pure_decompostion=False)    # 문제 있음.
#
#

# config_path = "/Users/jth_mac/Desktop/git/acl_tokenization/resources/mecab_fixed_decomposed_pure_sp-64k/tok.json"
# mc = MeCabTokenizer(config_path=config_path)

# text = "사람은 널 원해.\n"
# text = "사람은 너를 원해.\n아파르트헤이트는 큰 문제였다.\n"
# text = "사람은 너를 원해.\n너를 죽이겠다.\n"
#
#
# mc.tokenize(text)   # ['ㅅㅏ⊸ㄹㅏㅁ', 'ㅇㅡㄴ', '▃', 'ㄴㅓㄹ', '▃', 'ㅇㅝㄴㅎㅐ⊸', '.']
# mc.detokenize( mc.tokenize(text))