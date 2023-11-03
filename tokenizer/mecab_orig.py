#  mecab_orig
# kortok API 이용해 자모 분해까지 덧붙인 버전

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

            # cjj_ = ''.join(c if c != ' ' else self.dummy_letter for c in cjj)
            # return cjj_


    text_ = []
    for char in text:
        if character_is_korean(char):
            text_.append(transform_grammatical(char, grammatical=grammatical))
        else:
            text_.append(char)
    text_ = doublespace_pattern.sub(' ', ''.join(text_))
    return text_




class MeCabTokenizer_orig(BaseTokenizer):
    def __init__(self, tokenizer_type: str, decomposition_type: str, space_symbol: str = "▃", dummy_letter: str = ""):
        assert (tokenizer_type in ["mecab_orig", "mecab_fixed"] ), 'check the tokenizer type!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        assert (decomposition_type in ["composed", "decomposed_pure", "decomposed_morphological"] ), 'check the decomposition type!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

        self.mecab = MeCab.Tagger(f"--dicdir /usr/local/lib/mecab/dic/mecab-ko-dic")
        # self.use_original = use_original    # True: mecab orig  False: mecab fixed
        self.tokenizer_type = tokenizer_type  # mecab_orig  / mecab_fixed

        self.decomposition_type = decomposition_type    # composed  decomposed_pure  decomposed_morphological
        self.space_symbol = space_symbol    # 단어 사이 특수 문자
        self.dummy_letter = dummy_letter    # 초성/중성/종성 자리 채우기용 더미 문자

        self.grammatical_pos = ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC", "EP", "EF", "EC", "ETN", "ETM"]    # 어미, 조사

        self.tok = tok.tokenizers(dummy_letter=self.dummy_letter , space_symbol=self.space_symbol)


    # kortok API based
    def tokenize(self, text: str) -> List[str]:
        text = text.strip()
        text_ptr = 0    # 3
        tokenized = []  # ['나', 'ᆫ', '▃', '너', 'ᆯ']
        for ix in range(len(self.mecab.parse(text).split("\n"))) :
        # for ix in range(0,2 ):
            mor = self.mecab.parse(text).split("\n")[ix]

            if "\t" in mor:
                splitted = mor.split("\t") # 형태소 토큰과 나머지 부분 분리  # '난\t', 'NP+JX,*,T,난,Inflect,NP,JX,나/NP/*+ᆫ/JX/*'
                token = splitted[0] # 형태소 토큰    # '난\t'
                pos = splitted[1].split(",", 1)[0]

                if text[text_ptr] == " ":   # 현재 인덱스(text_ptr) character 가 스페이스라면
                    while text[text_ptr] == " ":    # 스페이스(띄어쓰기) 나타나는 부분까지 인덱스(text_ptr) 이동시킨 후 space symbol 삽입
                        text_ptr += 1
                    assert (
                            text[text_ptr] == token[0]
                    ), f"{repr(text)}//{text_ptr}//{text[text_ptr]}//{token}//{token[0]}\n"

                    tokenized.append(self.space_symbol)

                # tokenized.append(token)  # 토큰화해서 결과 저장

                # if self.use_original == True:   # mecab original
                if self.tokenizer_type == "mecab_orig":  # mecab original
                    if self.decomposition_type == "composed":
                        tokenized.append(token)
                    elif self.decomposition_type == "decomposed_pure":
                        tokenized.append(str2jamo(token, grammatical=False, dummy_letter=self.dummy_letter))   # 자모 분해 후 추가
                    elif self.decomposition_type == "decomposed_morphological":
                        if sum([1 for pos in pos.split("+") if pos in self.grammatical_pos]) < 1:  # VV+EC 등 고려해도 문법 형태소 없으면
                            tokenized.append(token) # 그대로 추가
                        elif sum([1 for pos in pos.split("+") if pos in self.grammatical_pos]) >= 1:  # VV+EC 등 고려해서 문법 형태소 있으면
                            tokenized.append(str2jamo(token, grammatical=False, dummy_letter=self.dummy_letter))   # 자모 분해 후 추가

                text_ptr += len(token)

        return tokenized



    # orig composed에 대해서만 작동
    def detokenize(self, tokens: List[str]) -> str:
        text = "".join(tokens).replace("▃", " ").strip()
        return text










# config_path = "./resources/v5_with_dummy_letter/mecab_fixed_decomposed_morphological_sp-64k/tok.json"
# decomposition_type = "composed"
# decomposition_type = "decomposed_pure"
# mc = MeCabTokenizer_orig(tokenizer_type="mecab_orig", decomposition_type="composed")                    # ['사람', '은', '▃', '널', '▃', '진짜', '▃', '원해', '.']
# mc = MeCabTokenizer_orig(tokenizer_type="mecab_orig", decomposition_type="decomposed_pure")             # ['ㅅㅏㄹㅏㅁ', 'ㅇㅡㄴ', '▃', 'ㄴㅓㄹ', '▃', 'ㅈㅣㄴㅉㅏ', '▃', 'ㅇㅝㄴㅎㅐ', '.']
# mc = MeCabTokenizer_orig(tokenizer_type="mecab_orig", decomposition_type="decomposed_morphological")             # ['사람', 'ㅇㅡㄴ', '▃', 'ㄴㅓㄹ', '▃', '진짜', '▃', 'ㅇㅝㄴㅎㅐ', '.']
#
# mc = MeCabTokenizer_fixed(use_original=True, decomposition_type="decomposed_morphological")    # ['사람', 'ㅇㅡㄴ', '▃', 'ㄴㅓㄹ', '▃', '진짜', '▃', 'ㅇㅝㄴㅎㅐ', '.']
# mc = MeCabTokenizer_fixed(use_original=False, decomposition_type="composed")                   # ['사람', '은', '▃', '너', 'ᆯ', '▃', '진짜', '▃', '원하', '아', '.']
# mc = MeCabTokenizer_fixed(use_original=False, decomposition_type="decomposed_pure")            # ['ㅅㅏㄹㅏㅁ', 'ㅇㅡㄴ', '▃', 'ㄴㅓ', 'ᆯ', '▃', 'ㅈㅣㄴㅉㅏ', '▃', 'ㅇㅝㄴㅎㅏ', 'ㅇㅏ', '.']
# mc = MeCabTokenizer_fixed(use_original=False, decomposition_type="decomposed_morphological")   # ['사람', 'ㅇㅡㄴ', '▃', '너', 'ᆯ', '▃', '진짜', '▃', '원하', 'ㅇㅏ', '.']
#
# mc.tokenize("사람은 널 진짜 원해.")   #
#
#
# sent = "강남 비타 에듀 학원에 다닌다"
# sent = "이번에 캘리 중위는 전역한다"
# sent = "오늘의 내셔날 지오그래픽은 재밌다"
# sent = "어디서 콜라비 좀 사 와"
# sent = "들어간다"
# sent = "넌 들어간다"
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
