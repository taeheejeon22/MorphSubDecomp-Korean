# mecab fixed
# konlpy 버리고 kortok 방식 따라 하기


import json
import os
import re
from typing import List

import MeCab

from tokenizer.base import BaseTokenizer


regexp = re.compile(".+(?=/[^A-Z])") # a pattern for only morphemes and their POS (e.g. 불태워/VV/* > 불태워/VV)


def split(elem, join=False):
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





class MeCabTokenizer_kortok(BaseTokenizer):
    def __init__(self, config_path: str):
        self.mecab = MeCab.Tagger(f"--dicdir /usr/local/lib/mecab/dic/mecab-ko-dic")
        with open(config_path) as f:
            self.config: dict = json.load(f)

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
                # pos = splitted[1].split(",", 1)[0]

                if text[text_ptr] == " ":   # 현재 인덱스(text_ptr) character 가 스페이스라면
                    while text[text_ptr] == " ":    # 스페이스(띄어쓰기) 나타나는 부분까지 인덱스(text_ptr) 이동시킨 후 space symbol 삽입
                        text_ptr += 1
                    assert (
                            text[text_ptr] == token[0]
                    ), f"{repr(text)}//{text_ptr}//{text[text_ptr]}//{token}//{token[0]}\n"

                    tokenized.append(self.config["space_symbol"])

                # tokenized.append(token)  # 토큰화해서 결과 저장
                tokenized += [mor_pos[0] for mor_pos in split(mor)]  # ['나', 'ᆫ'] 진짜 형태소로 쪼개진 토큰들 저장
                text_ptr += len(token)

        return tokenized

    def detokenize(self, tokens: List[str]) -> str:
        text = "".join(tokens).replace("▃", " ").strip()
        return text



# config_path = "./resources/v2_with_dummy_letter/wikiko_all_64k/mecab_fixed_decomposed_morphological_sp-64k/tok.json"
# mc = MeCabTokenizer_kortok(config_path)
#
# self = mc
#
#
# mc.tokenize("사람은 너를 원해.\n아파르트헤이트는 큰 문제였다.\n너를 죽이겠다.")
# mc.tokenize("사람은 너를 원해.\n아파르트헤이트는 큰 문제였다.\n")
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