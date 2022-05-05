from abc import abstractmethod# mc = MeCabTokenizer_fixed(tokenizer_type="mecab_orig", decomposition_type="composed", space_symbol= "▃", dummy_letter= "" )                    # ['사람', '은', '▃', '널', '▃', '진짜', '▃', '원해', '.']
# mc = MeCabTokenizer_fixed(tokenizer_type="mecab_orig", decomposition_type="decomposed_pure", space_symbol= "▃", dummy_letter= "" )
# mc = MeCabTokenizer_fixed(tokenizer_type="mecab_orig", decomposition_type="decomposed_morphological", space_symbol= "▃", dummy_letter= "" )
# mc = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="composed", space_symbol= "▃", dummy_letter= "" )                    # ['사람', '은', '▃', '널', '▃', '진짜', '▃', '원해', '.']
# mc = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure", space_symbol= "▃", dummy_letter= "" )
# mc = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="decomposed_morphological", space_symbol= "▃", dummy_letter= "" )
#
# mc = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="decomposed_morphological", space_symbol= "▃", dummy_letter= "⊸" )   # ['나', '⊸⊸ㄴ', '▃', '너', '⊸⊸ㄹ', '▃', '진짜', '▃', '원하', 'ㅇㅏ⊸', '.']
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

from typing import List


class BaseTokenizer:
    """Tokenizer meta class"""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        raise NotImplementedError("Tokenizer::tokenize() is not implemented")
