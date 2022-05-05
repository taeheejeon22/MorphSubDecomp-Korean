# mecab all: wordpiece용. 아래 코드 변경
    # class MeCabTokenizer_all(BaseTokenizer):
    #     def __init__(self, token_type: str, tokenizer_type: str, decomposition_type: str, space_symbol: str = "", dummy_letter: str = "", nfd: bool = True, grammatical_symbol: list = ["", ""]):

# mecab orig, fixed 동시에 처리할 수 있는 토크나이저


# mecab_fixed_v2
# konlpy 방식. OS마다 결과 다른 거 땜에 어쩔 수 없이 내 버전 써야 됨.
# mecab.py + mecab_fixed.py 합친 것


# mecab fixed decomposition pure (추후 simple로 교체 예정)
# _init_에 추가하기, 아니면 걍 mecab_fixed에 통합하는 게 나을 듯.



import json
import os
import re
from typing import List

import MeCab

from soynlp.hangle import compose, decompose, character_is_korean, character_is_complete_korean, character_is_moum, character_is_jaum
from tokenizer.base import BaseTokenizer


# import scripts.tokenizers_acl_v2 as tok
# import scripts.tokenizers_acl_v3 as tok
import scripts.tokenizers_acl_v3_2 as tok   # LG (lexical grammatical) 기능 추가



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

            # cjj_ = ''.join(c if c != ' ' else self.dummy_letter for c in cjj)
            # return cjj_


        # if jamo_morpheme == False:
        #     text_ = []
        #     for char in text:
        #         if character_is_korean(char):
        #             text_.append(transform(char))
        #         else:
        #             text_.append(char)
        #     text_ = doublespace_pattern.sub(' ', ''.join(text_))
        #     return text_
        #
        # if jamo_morpheme == True:   # for jamo morphemes like ㄴ, ㄹ, ...
        #     return self.dummy_letter*2 + text   # '##ㄴ'

    text_ = []
    for char in text:
        if character_is_korean(char):
            text_.append(transform_grammatical(char, grammatical=grammatical))
        else:
            text_.append(char)
    text_ = doublespace_pattern.sub(' ', ''.join(text_))
    return text_


# # morphological decomposition
# def mecab_with_morphological_decomposition(sent, use_original, dummy_letter, space_symbol):
#     '''
#     :param sent: 자모 변환할 문장      '너를 좋아해'
#     :param morpheme_analysis:
#         False: 자모 변환만 수행    (어절 토큰화 문장을 자모로 변환하는 데에 그대로 이용 가능)
#         True: 형태소 분석 + 자모 변환
#     :param use_original: konlpy original mecab 쓸지
#     :return: 자모 변환된 문장          '너ㅡㄹ 좋아해' or '너 ㄹㅡㄹ 좋아해'
#     '''
#
#     # 음절 분해용: 난 > ㄴㅏㄴ
#     # https://github.com/ratsgo/embedding/blob/master/preprocess/unsupervised_nlputils.py
#     def transform_v2(char):
#         if char == ' ':  # 공백은 그대로 출력
#             return char
#
#         cjj = decompose(char)  # soynlp 이용해 분해
#
#         # 자모 하나만 나오는 경우 처리 # ㄴ ㅠ
#         try:
#             if cjj.count(" ") == 2:
#                 if character_is_jaum(cjj[0]):  # 그 자모가 자음이면
#                     cjj = (dummy_letter, dummy_letter, cjj[0])  # ('ㄴ', ' ', ' ') > ('-', 'ㄴ', '-')
#                 elif character_is_moum(cjj[0]):  # 그 자모가 모음이면
#                     cjj = (dummy_letter, cjj[1], dummy_letter)  # (' ', 'ㅠ', ' ') > ('-', 'ㅠ', '-')
#         except AttributeError:  # 혹시라도 한글 아닌 것이 들어올 경우 대비해
#             pass
#
#         if len(cjj) == 1:
#             return cjj
#
#         cjj_ = ''.join(c if c != ' ' else dummy_letter for c in cjj)
#         return cjj_
#
#
#
#     if use_original == True:
#         mors_ejs_in_sent = mc_orig.pos(sent, flatten=False)  # 형태소 분석
#     elif use_original == False:
#         mors_ejs_in_sent = mc_fixed.pos(sent, flatten=False)  # 형태소 분석
#
#
#
#
#
#
#
#     new_sent = list()
#     for ix in range(len(mors_ejs_in_sent)):
#         eojeol = mors_ejs_in_sent[ix]  # [('나', 'NP'), ('는', 'JX')]
#
#         new_eojeol = list()  # ['나', 'ㄴㅡㄴ']
#         for jx in range(len(eojeol)):
#             morpheme, pos = eojeol[jx]  # '너', 'NP'
#
#             # 문법 형태소가 아니면
#             # if not pos in grammatical_pos:    # 잔다 VV+EC 등을 분해하지 않음
#             if sum([1 for pos in pos.split("+") if pos in grammatical_pos]) < 1:  # 잔다 VV+EC 등을 분해함
#                 decomposed_morpheme = morpheme[:]
#
#             # 문법 형태소이면
#             # elif pos in grammatical_pos:  # 잔다 VV+EC 등을 분해하지 않음
#             elif sum([1 for pos in pos.split("+") if pos in grammatical_pos]) >= 1:  # 잔다 VV+EC 등을 분해함
#                 decomposed_morpheme = "".join(
#                     [transform_v2(char) if character_is_korean(char) else char for char in morpheme])
#
#             new_eojeol.append(decomposed_morpheme)
#
#         new_sent.append(new_eojeol)
#
#         # if morpheme_tokenization == False:  # 형태소 토큰화 없이 어절 그대로 자모로 변환만 한다면
#         #     # if flatten == True:
#         #     #     new_sent.append("".join(new_eojeol))
#         #     # elif flatten == False:
#         #     new_sent.append(new_eojeol)
#         # elif morpheme_tokenization == True:  # 형태소 토큰화 + 자모 변환 한다면
#         #     # if flatten == True:
#         #     #     new_sent += new_eojeol
#         #     # elif flatten == False:
#         #     new_sent.append(new_eojeol)
#
#     # if flatten == True:
#     #     new_sent = doublespace_pattern.sub(" ", " ".join(new_sent))
#     # elif flatten == False:
#     #     pass
#
#     new_sent_with_special_token = list(chain.from_iterable(intersperse(new_sent, space_symbol)))
#
#     return new_sent_with_special_token






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


    # # kortok API based
    # def tokenize(self, text: str) -> List[str]:
    #     text = text.strip()
    #     text_ptr = 0    # 3
    #     tokenized = []  # ['나', 'ᆫ', '▃', '너', 'ᆯ']
    #     for ix in range(len(self.mecab.parse(text).split("\n"))) :
    #     # for ix in range(0,2 ):
    #         mor = self.mecab.parse(text).split("\n")[ix]
    #
    #         if "\t" in mor:
    #             splitted = mor.split("\t") # 형태소 토큰과 나머지 부분 분리  # '난\t', 'NP+JX,*,T,난,Inflect,NP,JX,나/NP/*+ᆫ/JX/*'
    #             token = splitted[0] # 형태소 토큰    # '난\t'
    #             pos = splitted[1].split(",", 1)[0]
    #
    #             if text[text_ptr] == " ":   # 현재 인덱스(text_ptr) character 가 스페이스라면
    #                 while text[text_ptr] == " ":    # 스페이스(띄어쓰기) 나타나는 부분까지 인덱스(text_ptr) 이동시킨 후 space symbol 삽입
    #                     text_ptr += 1
    #                 assert (
    #                         text[text_ptr] == token[0]
    #                 ), f"{repr(text)}//{text_ptr}//{text[text_ptr]}//{token}//{token[0]}\n"
    #
    #                 tokenized.append(self.space_symbol)
    #
    #             # tokenized.append(token)  # 토큰화해서 결과 저장
    #
    #             # if self.use_original == True:   # mecab original
    #             if self.tokenizer_type == "mecab_orig":  # mecab original
    #                 if self.decomposition_type == "composed":
    #                     tokenized.append(token)
    #                 elif self.decomposition_type == "decomposed_pure":
    #                     tokenized.append(str2jamo(token, grammatical=False, dummy_letter=self.dummy_letter))   # 자모 분해 후 추가
    #                 elif self.decomposition_type == "decomposed_morphological":
    #                     if sum([1 for pos in pos.split("+") if pos in self.grammatical_pos]) < 1:  # VV+EC 등 고려해도 문법 형태소 없으면
    #                         tokenized.append(token) # 그대로 추가
    #                     elif sum([1 for pos in pos.split("+") if pos in self.grammatical_pos]) >= 1:  # VV+EC 등 고려해서 문법 형태소 있으면
    #                         tokenized.append(str2jamo(token, grammatical=False, dummy_letter=self.dummy_letter))   # 자모 분해 후 추가
    #
    #             # elif self.use_original == False:    # mecab fixed
    #             elif self.tokenizer_type == "mecab_fixed":  # mecab fixed
    #                 if self.decomposition_type == "composed":
    #                     mecab_tokenized = [mor_pos[0] for mor_pos in mecab_tokenize(mor)]  # ['나', 'ᆫ'] 진짜 형태소로 쪼개진 토큰들 저장
    #                     tokenized += mecab_tokenized
    #                 elif self.decomposition_type == "decomposed_pure":
    #                     mecab_tokenized = [mor_pos[0] for mor_pos in mecab_tokenize(mor)]  # ['나', 'ᆫ'] 진짜 형태소로 쪼개진 토큰들 저장
    #                     tokenized += [str2jamo(token, grammatical=False, dummy_letter=self.dummy_letter) for token in mecab_tokenized] # 자모 분해 후 추가
    #                 elif self.decomposition_type == "decomposed_morphological":
    #                     mecab_tokenized_with_pos = mecab_tokenize(mor)[:]  # [('나', 'NP'), ('ᆫ', 'JX')] 진짜 형태소로 쪼개진 토큰들 저장 with POS tag
    #                     tokenized += [mor_pos[0] if (not mor_pos[-1] in self.grammatical_pos) else str2jamo(mor_pos[0], grammatical=False, dummy_letter=self.dummy_letter) for mor_pos in mecab_tokenized_with_pos]    # 어휘 형태소는 그대로, 문법 형태소는 자모 분해 후 추가
    #
    #             text_ptr += len(token)
    #
    #     return tokenized


    # our (konlpy based)
    def tokenize(self, text: str) -> List[str]:
        text = text.strip()

        # return self.tok.mecab_tokenizer(text, use_original=self.use_original, pure_decomposition=self.pure_decomposition, morphological=self.morphological)
        # return self.tok.mecab_tokenizer(text, token_type=self.token_type, tokenizer_type=self.tokenizer_type, decomposition_type=self.decomposition_type)

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
# mc = MeCabTokenizer_all(tokenizer_type="mecab_fixed", decomposition_type="composed", space_symbol= "▃", dummy_letter= "" )                    # ['사람', '은', '▃', '널', '▃', '진짜', '▃', '원해', '.']
# mc = MeCabTokenizer_all(tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure", space_symbol= "▃", dummy_letter= "" )
# mc = MeCabTokenizer_all(tokenizer_type="mecab_fixed", decomposition_type="decomposed_morphological", space_symbol= "▃", dummy_letter= "" )
#
# mc = MeCabTokenizer_all(tokenizer_type="mecab_fixed", decomposition_type="decomposed_morphological", space_symbol= "▃", dummy_letter= "⊸" )   # ['나', '⊸⊸ㄴ', '▃', '너', '⊸⊸ㄹ', '▃', '진짜', '▃', '원하', 'ㅇㅏ⊸', '.']
#
#
#
# sent = "나는 널 먹는데."
# mc.tokenize(sent)
#
# len(mc.tokenize(sent)[0])
# len(mc.tokenize(sent)[1])
#
#
#
# mc.tokenize("난 널 진짜 원해.")   # ['나', 'ㄴ', '▃', '너', 'ㄹ', '▃', '진짜', '▃', '원하', '아', '.']
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
