# mecab_fixed_sp, mecab_orig_sp 통합하고, mecab_all도 처리할 수 있도록 함.

# mecab orig / fixed / all 선택
# use_fixed: bool

from typing import List

from tokenizer.base import BaseTokenizer

# from tokenizer.mecab import MeCabTokenizer
from tokenizer.mecab_orig import MeCabTokenizer_orig
from tokenizer.mecab_fixed import MeCabTokenizer_fixed
from tokenizer.mecab_all import MeCabTokenizer_all

from tokenizer.sentencepiece import SentencePieceTokenizer


class MeCabSentencePieceTokenizer(BaseTokenizer):
    def __init__(self, mecab, sp: SentencePieceTokenizer):
        self.mecab = mecab
        self.sp = sp
        # self.use_fixed = use_fixed

        # self.mecab = MeCabTokenizer(config_path="./resources/mecab_orig_composed_sp-64k/tok.json")
        # self.sp = SentencePieceTokenizer(model_path="./resources/mecab_orig_composed_sp-64k/tok.model")
        # self.mecab = MeCabTokenizer(config_path="./resources/mecab_fixed_composed_sp-64k/tok.json")


        # self.mecab = MeCabTokenizer_orig(tokenizer_type="mecab_orig", decomposition_type="composed")
        # self.sp = SentencePieceTokenizer(model_path="./resources/v5_without_dummy_letter/mecab_orig_composed_sp-64k/tok.model")
        # self.mecab = MeCabTokenizer_orig(tokenizer_type="mecab_orig", decomposition_type="decomposed_morphological")
        # self.sp = SentencePieceTokenizer(model_path="./resources/v5_without_dummy_letter/mecab_orig_decomposed_morphological_sp-64k/tok.model")

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





# # mecab_orig.py
# mecab = MeCabTokenizer_orig(tokenizer_type="mecab_orig", decomposition_type="composed")
# sp = SentencePieceTokenizer(model_path="./resources/v5_without_dummy_letter/mecab_orig_composed_sp-64k/tok.model")
# mecab = MeCabTokenizer_orig(tokenizer_type="mecab_orig", decomposition_type="decomposed_pure")
# sp = SentencePieceTokenizer(model_path="./resources/v5_without_dummy_letter/mecab_orig_decomposed_pure_sp-64k/tok.model")
# mecab = MeCabTokenizer_orig(tokenizer_type="mecab_orig", decomposition_type="decomposed_morphological") # ['▁나', '▁ㄴㅡㄴ', '▃', '▁장', '풍', '▁ㅇㅡㄹ', '▃', '▁ㅎㅐㅆ', '▁ㄷㅏ', '▁.']
# sp = SentencePieceTokenizer(model_path="./resources/v5_without_dummy_letter/mecab_orig_decomposed_morphological_sp-64k/tok.model")
#
#
# # mecab_fixed.py
# mecab = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="composed", space_symbol= "▃", dummy_letter= "" ) # ['▁나', '▁는', '▃', '▁장풍', '▁을', '▃', '▁하', '▁았', '▁다', '▁.']
# sp = SentencePieceTokenizer(model_path="./resources/v5_without_dummy_letter/mecab_fixed_composed_sp-64k/tok.model")
# mecab = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure", space_symbol= "▃", dummy_letter= "" )
# sp = SentencePieceTokenizer(model_path="./resources/v5_without_dummy_letter/mecab_fixed_decomposed_pure_sp-64k/tok.model")
# mecab = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="decomposed_morphological", space_symbol= "▃", dummy_letter= "" )
# sp = SentencePieceTokenizer(model_path="./resources/v5_without_dummy_letter/mecab_fixed_decomposed_morphological_sp-64k/tok.model")
#
#
# # mecab_all.py
# mecab = MeCabTokenizer_all(token_type="eojeol", tokenizer_type="mecab_fixed", decomposition_type="composed")    # ['▁전태', '희는', '▁한국', '대학교에', '▁묵', '었었다']
# sp = SentencePieceTokenizer(model_path="./resources/v6_without_dummy_letter/eojeol_mecab_fixed_composed_sp-64k/tok.model")
# mecab = MeCabTokenizer_all(token_type="eojeol", tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure_nfd") # ['▁전태', '희는', '▁한국', '대학교에', '▁묵', '어', 'ᆻ었다']
# sp = SentencePieceTokenizer(model_path="./resources/v6_without_dummy_letter/eojeol_mecab_fixed_decomposed_pure_nfd_sp-64k/tok.model")
# mecab = MeCabTokenizer_all(token_type="eojeol", tokenizer_type="mecab_fixed", decomposition_type="decomposed_morphological_nfd") # ['▁전태', '희는', '▁한국', '대학교에', '▁묵', '었었다']
# sp = SentencePieceTokenizer(model_path="./resources/v6_without_dummy_letter/eojeol_mecab_fixed_decomposed_morphological_nfd_sp-64k/tok.model")
#
# mecab = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="composed")    # ['▁전태', '희', '▁는', '▃', '▁한국', '▁대학교', '▁에', '▃', '▁묵', '▁었었', '▁다']     # ['▁난', '▃', '▁널', '▃', '▁좋아해']
# sp = SentencePieceTokenizer(model_path="./resources/v6_without_dummy_letter/morpheme_mecab_orig_composed_sp-64k/tok.model")
# mecab = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="decomposed_pure_nfd") # ['▁전태', '희', '▁는', '▃', '▁한국', '▁대학교', '▁에', '▃', '▁묵', '▁었었', '▁다']
# sp = SentencePieceTokenizer(model_path="./resources/v6_without_dummy_letter/morpheme_mecab_orig_decomposed_pure_nfd_sp-64k/tok.model")
# mecab = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="decomposed_morphological_nfd") # ['▁전태', '희', '▁는', '▃', '▁한국', '▁대학교', '▁에', '▃', '▁묵', '▁었었', '▁다']
# sp = SentencePieceTokenizer(model_path="./resources/v6_without_dummy_letter/morpheme_mecab_orig_decomposed_morphological_nfd_sp-64k/tok.model")
#
# mecab = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="composed_nfd", space_symbol="")    # ['▁전태', '희', '▁는', '▃', '▁한국', '▁대학교', '▁에', '▃', '▁묵', '▁었었', '▁다']    # ['▁나', '▁ᆫ', '▃', '▁너', '▁ᆯ', '▃', '▁좋아하', '▁아']
# sp = SentencePieceTokenizer(model_path="./resources/v6_without_dummy_letter/morpheme_mecab_fixed_composed_nfd_sp-64k/tok.model")
# mecab = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure_nfd") # ['▁전태', '희', '▁는', '▃', '▁한국', '▁대학교', '▁에', '▃', '▁묵', '▁었었', '▁다']
# sp = SentencePieceTokenizer(model_path="./resources/v6_without_dummy_letter/morpheme_mecab_fixed_decomposed_pure_nfd_sp-64k/tok.model")
# mecab = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="decomposed_morphological_nfd") # ['▁전태', '희', '▁는', '▃', '▁한국', '▁대학교', '▁에', '▃', '▁묵', '▁었었', '▁다']
# sp = SentencePieceTokenizer(model_path="./resources/v6_without_dummy_letter/morpheme_mecab_fixed_decomposed_morphological_nfd_sp-64k/tok.model")
#
#
#
# # sentence: eojeol / decomposed_morphological_nfd       # vocab: morpheme / decomposed_morphological_nfd
# mecab = MeCabTokenizer_all(token_type="eojeol", tokenizer_type="mecab_fixed", decomposition_type="decomposed_morphological_nfd")    # ['▁전태', '희', '는', '▁한국', '대', '학교', '에', '▁묵', '었', '었', '다']
# sp = SentencePieceTokenizer(model_path="./resources/v6_without_dummy_letter/morpheme_mecab_fixed_decomposed_morphological_nfd_sp-64k/tok.model")
#
#
#
#
# # mc-sp
#
# mc = MeCabSentencePieceTokenizer(mecab=mecab, sp=sp)
# mecab.tokenize(text)
# mc.tokenize(text)
#
#
#
# # text = '나는 오늘 저녁을 먹었다.'   # ['▁나', '▁는', '▃', '▁오늘', '▃', '▁저녁', '▁을', '▃', '▁먹', '▁', '었', '▁다', '▁.']
# # text = "대한민국에 우리끼리 살아보자"    # ['▁대한민국', '▁에', '▃', '▁우리', '▁끼', '리', '▃', '▁살', '▁아', '▁보', '▁자']
# # text = "사망 플래그의 좋은 예시이다."
# text = "나는 장풍을 했다."
# text = "전태희는 한국대학교에 묵었었다"
# text = "나는 장소를 했다."
# text = "난 널 좋아해"
#
#
#
# len(mc.tokenize(text)[0])
# len(mc.tokenize(text)[1])
# len(mc.tokenize(text)[2])



# self = mc
# self.tokenize(text)
