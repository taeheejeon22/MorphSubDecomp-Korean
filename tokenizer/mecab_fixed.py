# mecab fixed

import json
import os
import re
from typing import List

import MeCab

from soynlp.hangle import compose, decompose, character_is_korean, character_is_complete_korean, character_is_moum, character_is_jaum
from tokenizer.base import BaseTokenizer

import scripts.tokenizer_collection as tok



class MeCabTokenizer_fixed(BaseTokenizer):
    # def __init__(self, tokenizer_type: str, decomposition_type: str, space_symbol: str = "▃", dummy_letter: str = ""):
    def __init__(self, tokenizer_type: str, decomposition_type: str, space_symbol: str = "▃", dummy_letter: str = "", token_type: str ="morpheme", lexical_grammatical: bool = False):
        assert (tokenizer_type in ["mecab_orig", "mecab_fixed"] ), 'check the tokenizer type!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        assert (decomposition_type in ["composed", "decomposed_pure", "decomposed_morphological", "decomposed_lexical", "decomposed_grammatical"] ), 'check the decomposition type!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

        self.mecab = MeCab.Tagger(f"--dicdir /usr/local/lib/mecab/dic/mecab-ko-dic")
        # self.use_original = use_original    # True: mecab orig  False: mecab fixed
        self.tokenizer_type = tokenizer_type  # mecab_orig  / mecab_fixed

        self.lexical_grammatical = lexical_grammatical  # LG 적용 여부 (내셔널 지오 그래픽 vs. 내셔널지오그래픽)

        self.token_type = token_type
        self.decomposition_type = decomposition_type    # composed  decomposed_pure  decomposed_morphological
        self.space_symbol = space_symbol    # 단어 사이 특수 문자
        self.dummy_letter = dummy_letter    # 초성/중성/종성 자리 채우기용 더미 문자

        self.grammatical_pos = ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC", "EP", "EF", "EC", "ETN", "ETM"]    # 어미, 조사

        self.tok = tok.tokenizers(dummy_letter=self.dummy_letter , space_symbol=self.space_symbol)



    # our (konlpy based)
    def tokenize(self, text: str) -> List[str]:
        text = text.strip()

        tokenizer = self.tok.mecab_tokenizer(text, tokenizer_type=self.tokenizer_type, token_type=self.token_type, decomposition_type=self.decomposition_type, lexical_grammatical=self.lexical_grammatical)

        return tokenizer



    # orig composed에 대해서만 작동
    def detokenize(self, tokens: List[str]) -> str:
        text = "".join(tokens).replace("▃", " ").strip()
        return text











# mc = MeCabTokenizer_fixed(tokenizer_type="mecab_orig", decomposition_type="decomposed_pure", space_symbol= "▃", dummy_letter= "" )
# mc = MeCabTokenizer_fixed(tokenizer_type="mecab_orig", decomposition_type="composed", space_symbol= "▃", dummy_letter= "" )                    # ['사람', '은', '▃', '널', '▃', '진짜', '▃', '원해', '.']
# # mc = MeCabTokenizer_fixed(tokenizer_type="mecab_orig", decomposition_type="decomposed_morphological", space_symbol= "▃", dummy_letter= "" )
# mc = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="composed", space_symbol= "▃", dummy_letter= "" )                    # ['사람', '은', '▃', '널', '▃', '진짜', '▃', '원해', '.']
# mc = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure", space_symbol= "▃", dummy_letter= "" )
# # mc = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="decomposed_morphological", space_symbol= "▃", dummy_letter= "" )
# mc = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="decomposed_lexical", space_symbol= "▃", dummy_letter= "" )
# mc = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="decomposed_grammatical", space_symbol= "▃", dummy_letter= "" )
#
# mc = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="composed", space_symbol= "▃", dummy_letter= "", lexical_grammatical=True)
# mc = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure", space_symbol= "▃", dummy_letter= "", lexical_grammatical=True)
# # mc = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="decomposed_morphological", space_symbol= "▃", dummy_letter= "" )
# mc = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="decomposed_lexical", space_symbol= "▃", dummy_letter= "", lexical_grammatical=True)
# mc = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="decomposed_grammatical", space_symbol= "▃", dummy_letter= "", lexical_grammatical=True)
#
#
#
#
# mc.tokenize("난 널 진짜 원해.")   # ['나', 'ㄴ', '▃', '너', 'ㄹ', '▃', '진짜', '▃', '원하', '아', '.']
# mc.tokenize("나는 너를 먹는데.")
#
#
# sent = "강남 비타 에듀 학원에 다닌다"
# sent = "이번에 캘리 중위는 전역한다"
# sent = "오늘의 내셔날 지오그래픽은 재밌다"
# sent = "어디서 콜라비 좀 사 와"
# sent = "들어간다"
# sent = "넌 들어간다"
# sent = "발생하는 한국고려대학교에서는 빨리는 먹지는 못해서요."
# sent = "내셔날 지오그래픽은 재밌다"
#
# mc.tokenize(sent)
#
# self = mc
#
# mc.tokenize("사람은 너를 원해.")
# mc.tokenize("사람은 너를 원해.\n아파르트헤이트는 큰 문제였다.\n너를 죽이겠다.")
# mc.tokenize("사람은 너를 원해.\n아파르트헤이트는 큰 문제였다.\n")
#
#
# text = "사람은 너를 원해.\n"
# text = "사람은 너를 원해.\n아파르트헤이트는 큰 문제였다.\n"
# text = "사람은 너를 원해.\n너를 죽이겠다.\n"
# text = "난 널 진짜 원한다"
# text = "이것이 아니다"
# text = "재밌음ㅋㅋ"
# text = "재밌음ㅠㅠ"
# text = "넌 날 좋아해"
# text = "미궁에서 뜬 아앗"
# text = "훌륭한 사망 플래그의 예시이다"
# text = "수해에 입장한다"   # ['ㅅㅜ#ㅎㅐ#', 'ㅇㅔ#', '▃', 'ㅇㅣㅂㅈㅏㅇ', 'ㅎㅏ#', 'ㄴ##ㄷㅏ#']
# text = "난 널 좋아해"
# text = '정답입니다.'
#
# mc.tokenize(text)   # ['사람', '은', '너', '를', '원해', '.']
