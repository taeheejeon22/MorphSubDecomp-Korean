# mecab orig / fixed 선택
# use_fixed: bool

from typing import List

from tokenizer.base import BaseTokenizer

# from tokenizer.mecab import MeCabTokenizer
from tokenizer.mecab_fixed import MeCabTokenizer_fixed

from tokenizer.sentencepiece import SentencePieceTokenizer


class MeCabSentencePieceTokenizer_fixed(BaseTokenizer):
    def __init__(self, mecab: MeCabTokenizer_fixed, sp: SentencePieceTokenizer, use_fixed: bool):
        self.mecab = mecab
        self.sp = sp
        self.use_fixed = use_fixed

        # self.mecab = MeCabTokenizer_fixed(config_path="./resources/mecab_orig_composed_sp-64k/tok.json")
        # self.sp = MeCabSentencePieceTokenizer_fixed(model_path="./resources/mecab_orig_composed_sp-64k/tok.model")
        # self.mecab = MeCabTokenizer(config_path="./resources/mecab_fixed_composed_sp-64k/tok.json")
        # self.sp = SentencePieceTokenizer(model_path="./resources/mecab_fixed_composed_sp-64k/tok.model")


        # self.mecab = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="composed", space_symbol= "▃", dummy_letter= "" )                    # ['사람', '은', '▃', '널', '▃', '진짜', '▃', '원해', '.']
        # self.sp = SentencePieceTokenizer(model_path="./resources/v5_without_dummy_letter/mecab_fixed_composed_sp-64k/tok.model")
        # self.mecab = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure", space_symbol= "▃", dummy_letter= "" )
        # self.sp = SentencePieceTokenizer(model_path="./resources/v5_without_dummy_letter/mecab_fixed_decomposed_pure_sp-64k/tok.model")
        # self.mecab = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="decomposed_morphological", space_symbol= "▃", dummy_letter= "" )
        # self.sp = SentencePieceTokenizer(model_path="./resources/v5_without_dummy_letter/mecab_fixed_decomposed_morphological_sp-64k/tok.model")


    def tokenize(self, text: str) -> List[str]:
        # if self.use_fixed == False: # kortok API based tokenizer
        #     tokenized = self.mecab.tokenize(text)   # ['나', 'ᆫ', '▃', '너', 'ᆯ', '▃', '좋아하', '아']
        # elif self.use_fixed == True:  # our tokenizer (konlpy based)
        #     tokenized = self.mecab.tokenize(text)  # ['나', 'ᆫ', '▃', '너', 'ᆯ', '▃', '좋아하', '아']

        tokenized = self.mecab.tokenize(text)  # ['나', 'ᆫ', '▃', '너', 'ᆯ', '▃', '좋아하', '아']

        tokenized = self.sp.tokenize(" ".join(tokenized))   # ['▁나', '▁ㄴ', '▁▃', '▁너', '▁ㄹ', '▁▃', '▁좋아하', '▁아']

        output = []
        for i in range(0, len(tokenized)):
            if i + 1 < len(tokenized) and (tokenized[i] == "▁" and tokenized[i + 1] == "▃"):
                continue
            if tokenized[i] == "▁▃":    # 단어 사이 공백이면
                tokenized[i] = "▃"      # SentencePiece의 단어 시작 표지 제거
            output.append(tokenized[i])

        return output   # ['▁나', '▁ㄴ', '▃', '▁너', '▁ㄹ', '▃', '▁좋아하', '▁아']

    def detokenize(self, tokens: List[str]) -> str:
        text = "".join(tokens).replace("▁", "").replace(" ", "").replace("▃", " ").strip()
        return text







# mc = MeCabSentencePieceTokenizer_fixed(mecab=MeCabTokenizer_fixed, sp=SentencePieceTokenizer, use_fixed=True)
# text = '나는 오늘 저녁을 먹었다.'   # ['▁나', '▁는', '▃', '▁오늘', '▃', '▁저녁', '▁을', '▃', '▁먹', '▁', '었', '▁다', '▁.']
# text = "대한민국에 우리끼리 살아보자"    # ['▁대한민국', '▁에', '▃', '▁우리', '▁끼', '리', '▃', '▁살', '▁아', '▁보', '▁자']
# text = "사망 플래그의 좋은 예시이다."
# text = "나는 장풍을 했다."
# text = "난 널 좋아해"  # ['▁난', '▃', '▁널', '▃', '▁좋아해']
# mc.tokenize(text)

# self = mc
# self.tokenize(text)





# ['▁나', '▁는', '▁', '▃', '▁오늘', '▁', '▃', '▁저녁', '▁을', '▁', '▃', '▁먹', '▁', '었', '▁다', '▁.']
# ['▁나', '▁는', '▃', '▁오늘', '▃', '▁저녁', '▁을', '▃', '▁먹', '▁', '었', '▁다', '▁.']
# ['▁나', '▁는', '▃', '▁오늘', '▃', '▁저녁', '▁을', '▃', '▁먹', '▁었', '▁다', '▁.']
# 해결!