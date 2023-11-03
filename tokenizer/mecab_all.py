# mecab all: wordpiece용. mecab orig, fixed 동시에 처리할 수 있는 토크나이저.
    # 아래 코드 변경
    # class MeCabTokenizer_all(BaseTokenizer):
    #     def __init__(self, token_type: str, tokenizer_type: str, decomposition_type: str, space_symbol: str = "", dummy_letter: str = "", nfd: bool = True, grammatical_symbol: list = ["", ""]):

import json
import os
import re
from typing import List

import MeCab

from soynlp.hangle import compose, decompose, character_is_korean, character_is_complete_korean, character_is_moum, character_is_jaum
from tokenizer.base import BaseTokenizer

import scripts.tokenizer_collection as tok



regexp = re.compile(".+(?=/[^A-Z])") # a pattern for only morphemes and their POS (e.g. 불태워/VV/* > 불태워/VV)
doublespace_pattern = re.compile('\s+')


def mecab_tokenize(elem, join=False):
    # elem: an analysed result of an eojeol (e.g. 뭔지 > 뭔지\tNP+VCP+EC,*,F,뭔지,Inflect,NP,EC,뭐/NP/*+이/VCP/*+ㄴ지/EC/*)

    if not elem: return ('', 'SY')

    s, t = elem.split(
        '\t')  # s: an eojeol (e.g. 위한)   # t: analysed resulf of an eojeol (e.g. VV+ETM,*,T,위한,Inflect,VV,ETM,위하/VV/*+ᆫ/ETM/*)
    token_pos = t.split(',')[0]  # original token POS of mecab-ko (e.g. 위한: VV+ETM)
    lst_morpos = t.split(',')[-1].split(
        "+")  # splitting the last attr (인덱스 표현) of 't' by morpheme (e.g. 위하/VV/*+ᆫ/ETM/* > ["위하/VV/*", "ᆫ/ETM/*"])

    if join:
        if not t.split(',')[4].startswith(
                "Inflect"):  # If an eojeol is not Inflect (= a concatenation of morphemes is equal to its original eojeol. e.g. 해수욕장 == 해수 + 욕 + 장)
            return s + '/' + token_pos  # eojeol + / + POS (e.g. 위한/VV+ETM)
            # return [s + '/' + token_pos]  # eojeol + / + POS (e.g. 위한/VV+ETM)   # mecab_fixed에서는 걍 string으로 반환했었던 것 수정

        else:  # If an eojeol is Inflect (= a concatenation of morphemes is not equal to its original eojeol) (e.g. 불태워졌다 != 불태우 + 어 + 지 + 었 + 다)
            mor_info = [regexp.search(x).group() for x in
                        lst_morpos]  # make a list of morphemes with their POSs (e.g. ['줍/VV', '어서/EC'])

            # There is a bug that outputs of mecab-ko-dic are different according to OS, and OS versions. This is a make-shift.
            if len(mor_info) > 1:
                return mor_info
            elif len(mor_info) == 1:
                return [s + "/" + token_pos]
            # return [regexp.search(x).group() for x in lst_morpos]   # make a list of morphemes with their POSs (e.g. ['줍/VV', '어서/EC'] )

    else:
        if not t.split(',')[4].startswith("Inflect"):
            # return (s, token_pos)
            return [(s, token_pos)] # mecab_fixed에서는 걍 1차원으로 반환했었던 것 수정

        else:
            mor_info = [tuple(regexp.search(x).group().split("/")) for x in
                        lst_morpos]  # make a list of morphemes with their POSs (e.g. [('줍', 'VV'), ('어서', 'EC')] )

            # There is a bug that outputs of mecab-ko-dic are different according to OS, and OS versions. This is a make-shift.
            if len(mor_info) > 1:
                return mor_info
            elif len(mor_info) == 1:
                return (s, token_pos)


# mor_poss = split(mor)   # ['나/NP', 'ᆫ/JX']
#
#    " ".join([mor_pos.split("/")[0] for mor_pos in split(mor)])




# pure decomposition
def str2jamo(text, grammatical=False, dummy_letter=""):
    def transform(char):
        if char == ' ':
            return char
        cjj = decompose(char)
        if len(cjj) == 1:
            return cjj
        cjj_ = ''.join(c if c != ' ' else dummy_letter for c in cjj)
        return cjj_

    def transform_grammatical(char, grammatical):
        if char == ' ':
            return char
        cjj = decompose(char)
        if len(cjj) == 1:
            return cjj

        if grammatical == False:
            cjj_ = ''.join(c if c != ' ' else dummy_letter for c in cjj)
            return cjj_

        elif grammatical == True:
            cjj_without_blank = [x for x in cjj if x != " "] # remove " " from cjj

            if len(cjj_without_blank) == 1:   # if it is a jamo character (e.g. ㄴ, ㄹ, 'ㄴ'다)
                cjj_ = dummy_letter * 2 + cjj_without_blank[0]

            elif len(cjj_without_blank) != 1:   # if it is a syllable character (e.g. 은, 을, 는다)
                cjj_ = ''.join(c if c != ' ' else dummy_letter for c in cjj)

            return cjj_


    text_ = []
    for char in text:
        if character_is_korean(char):
            text_.append(transform_grammatical(char, grammatical=grammatical))
        else:
            text_.append(char)
    text_ = doublespace_pattern.sub(' ', ''.join(text_))
    return text_



class MeCabTokenizer_all(BaseTokenizer):
    # def __init__(self, token_type: str, tokenizer_type: str, decomposition_type: str, space_symbol: str = "", dummy_letter: str = "", nfd: bool = True, grammatical_symbol: list = ["", ""]):
    def __init__(self, token_type: str, tokenizer_type: str, decomposition_type: str, space_symbol: str = "", dummy_letter: str = "", nfd: bool = True, grammatical_symbol: list = ["", ""], lexical_grammatical: bool = False):   # for LG

        assert (token_type in ["eojeol", "morpheme"] ), 'check the token type!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        assert (tokenizer_type in ["mecab_orig", "mecab_fixed"] ), 'check the tokenizer type!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        # assert (decomposition_type in ["composed", "decomposed_pure", "decomposed_morphological", "composed_nfd", "decomposed_pure_nfd", "decomposed_morphological_nfd"] ), 'check the decomposition type!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

        self.mecab = MeCab.Tagger(f"--dicdir /usr/local/lib/mecab/dic/mecab-ko-dic")
        # self.use_original = use_original    # True: mecab orig  False: mecab fixed

        self.token_type = token_type    # eojeol / morpheme
        self.tokenizer_type = tokenizer_type  # mecab_orig  / mecab_fixed
        self.lexical_grammatical = lexical_grammatical  # LG 적용 여부 (내셔널 지오 그래픽 vs. 내셔널지오그래픽)

        self.decomposition_type = decomposition_type    # composed  decomposed_pure  decomposed_morphological
        self.space_symbol = space_symbol    # 단어 사이 특수 문자   # "▃"
        self.dummy_letter = dummy_letter    # 초성/중성/종성 자리 채우기용 더미 문자
        self.nfd = nfd  # NFD 이용해 자모 분해할지
        self.grammatical_symbol = grammatical_symbol    # 문법 형태소 표지

        self.grammatical_pos = ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC", "EP", "EF", "EC", "ETN", "ETM"]    # 어미, 조사

        self.tok = tok.tokenizers(dummy_letter=self.dummy_letter , space_symbol=self.space_symbol, nfd=self.nfd, grammatical_symbol=self.grammatical_symbol)    # 토크나이저 인스턴스 생성



    # our (konlpy based)
    def tokenize(self, text: str) -> List[str]:
        text = text.strip()

        tokenizer = self.tok.mecab_tokenizer(text, tokenizer_type=self.tokenizer_type, token_type=self.token_type, decomposition_type=self.decomposition_type, lexical_grammatical=self.lexical_grammatical)
        return tokenizer

    # orig composed에 대해서만 작동
    def detokenize(self, tokens: List[str]) -> str:
        text = "".join(tokens).replace("▃", " ").strip()
        return text











# mc = MeCabTokenizer_all(token_type="eojeol", tokenizer_type="mecab_orig", decomposition_type="composed", space_symbol= "▃", dummy_letter= "" )
# mc = MeCabTokenizer_all(token_type="eojeol", tokenizer_type="mecab_fixed", decomposition_type="composed", space_symbol= "▃", dummy_letter= "" )
# mc = MeCabTokenizer_all(token_type="eojeol", tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure", space_symbol= "▃", dummy_letter= "" )
# mc = MeCabTokenizer_all(token_type="eojeol", tokenizer_type="mecab_fixed", decomposition_type="decomposed_morphological_nfd", space_symbol= "▃", dummy_letter= "" )
#
#
# mc = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="composed", space_symbol= "▃", dummy_letter= "" )    # ['나', '는', '▃', '너', '를', '▃', '먹', '는데', '.']
# mc = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="decomposed_pure", space_symbol= "▃", dummy_letter= "" )
# mc = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="decomposed_morphological", space_symbol= "▃", dummy_letter= "" )
# mc = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="decomposed_pure_nfd", space_symbol= "▃", dummy_letter= "" )
# mc = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="decomposed_morphological_nfd", space_symbol= "▃", dummy_letter= "" )
#
# mc = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="composed", space_symbol= "▃", dummy_letter= "" )    # ['나', '는', '▃', '너', 'ㄹ', '▃', '먹', '는데', '.']
# mc = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure", space_symbol= "▃", dummy_letter= "" )
# mc = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="decomposed_morphological", space_symbol= "▃", dummy_letter= "" )
# mc = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure_nfd", space_symbol= "▃", dummy_letter= "" )
# mc = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="decomposed_morphological_nfd", space_symbol= "▃", dummy_letter= "" )
#
#
# mc = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="composed", space_symbol= "▃", dummy_letter= "" , lexical_grammatical=True)    # ['나', '는', '▃', '너', 'ㄹ', '▃', '먹', '는데', '.']
# mc = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure", space_symbol= "▃", dummy_letter= "", lexical_grammatical=True)
# mc = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="decomposed_lexical", space_symbol= "▃", dummy_letter= "", lexical_grammatical=True)
# mc = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="decomposed_grammatical", space_symbol= "▃", dummy_letter= "", lexical_grammatical=True)
#
#
# sent = "난 내셔날 지오그래픽은 좋았다"; print(mc.tokenize(sent))
#
#
# # mc = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="composed", space_symbol= "▃", dummy_letter= "" )   # ['나', '는', '▃', '너', 'ㄹ', '▃', '먹', '는데', '.']
#
#
#
#
# # mc = MeCabTokenizer_all(tokenizer_type="mecab_fixed", decomposition_type="composed", space_symbol= "▃", dummy_letter= "" )                    # ['사람', '은', '▃', '널', '▃', '진짜', '▃', '원해', '.']
# # mc = MeCabTokenizer_all(tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure", space_symbol= "▃", dummy_letter= "" )
# # mc = MeCabTokenizer_all(tokenizer_type="mecab_fixed", decomposition_type="decomposed_morphological", space_symbol= "▃", dummy_letter= "" )
# #
# # mc = MeCabTokenizer_all(tokenizer_type="mecab_fixed", decomposition_type="decomposed_morphological", space_symbol= "▃", dummy_letter= "⊸" )   # ['나', '⊸⊸ㄴ', '▃', '너', '⊸⊸ㄹ', '▃', '진짜', '▃', '원하', 'ㅇㅏ⊸', '.']
# #
# #
# #
# # sent = "나는 널 먹는데."
# # mc.tokenize(sent)
# #
# # len(mc.tokenize(sent)[0])
# # len(mc.tokenize(sent)[1])
# #
# #
# #
# # mc.tokenize("난 널 진짜 원해.")   # ['나', 'ㄴ', '▃', '너', 'ㄹ', '▃', '진짜', '▃', '원하', '아', '.']
# #
# # sent = "강남 비타 에듀 학원에 다닌다"
# # sent = "이번에 캘리 중위는 전역한다"
# # sent = "오늘의 내셔날 지오그래픽은 재밌다"
# # sent = "어디서 콜라비 좀 사 와"
# # sent = "들어간다"
# # sent = "넌 들어간다"
# # mc.tokenize(sent)
# #
# # self = mc
# #
# # mc.tokenize("사람은 너를 원해.")
# # mc.tokenize("사람은 너를 원해.\n아파르트헤이트는 큰 문제였다.\n너를 죽이겠다.")
# # mc.tokenize("사람은 너를 원해.\n아파르트헤이트는 큰 문제였다.\n")
# #
# #
# # text = "사람은 너를 원해.\n"
# # text = "사람은 너를 원해.\n아파르트헤이트는 큰 문제였다.\n"
# # text = "사람은 너를 원해.\n너를 죽이겠다.\n"
# # text = "난 널 진짜 원한다"
# # text = "이것이 아니다"
# # text = "재밌음ㅋㅋ"
# # text = "재밌음ㅠㅠ"
# # text = "넌 날 좋아해"
# # text = "미궁에서 뜬 아앗"
# # text = "훌륭한 사망 플래그의 예시이다"
# # text = "수해에 입장한다"   # ['ㅅㅜ#ㅎㅐ#', 'ㅇㅔ#', '▃', 'ㅇㅣㅂㅈㅏㅇ', 'ㅎㅏ#', 'ㄴ##ㄷㅏ#']
# # text = "난 널 좋아해"
# # text = '정답입니다.'
# #
# # mc.tokenize(text)   # ['사람', '은', '너', '를', '원해', '.']
