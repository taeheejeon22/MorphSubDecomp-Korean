# mecab_fixed_sp, mecab_orig_sp 통합하고, mecab_all도 처리할 수 있도록 함.

# mecab orig / fixed / all 선택
# use_fixed: bool

from typing import List

from tokenizer.base import BaseTokenizer

# from tokenizer.mecab import MeCabTokenizer
from tokenizer.mecab_orig import MeCabTokenizer_orig
from tokenizer.mecab_fixed import MeCabTokenizer_fixed
from tokenizer.mecab_all import MeCabTokenizer_all

from tokenizer.wordpiece import WordPieceTokenizer


class MeCabWordPieceTokenizer(BaseTokenizer):
    def __init__(self, mecab, wp: WordPieceTokenizer):
        self.mecab = mecab
        self.wp = wp
        # self.use_fixed = use_fixed


    def tokenize(self, text: str) -> List[str]:
        # if self.use_fixed == False: # kortok API based tokenizer
        #     tokenized = self.mecab.tokenize(text)   # ['나', 'ᆫ', '▃', '너', 'ᆯ', '▃', '좋아하', '아']
        # elif self.use_fixed == True:  # our tokenizer (konlpy based)
        #     tokenized = self.mecab.tokenize(text)  # ['나', 'ᆫ', '▃', '너', 'ᆯ', '▃', '좋아하', '아']

        tokenized = self.mecab.tokenize(text)  # ['나', 'ᆫ', '▃', '너', 'ᆯ', '▃', '좋아하', '아']     # ['내', '⭧가', '먹', '⭧다']

        tokenized = self.wp.tokenize(" ".join(tokenized))   # ['▁나', '▁ㄴ', '▁▃', '▁너', '▁ㄹ', '▁▃', '▁좋아하', '▁아']

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
# wp = WordPieceTokenizer(model_path="./resources/v5_without_dummy_letter/mecab_orig_composed_wp-64k/tok.model")
# mecab = MeCabTokenizer_orig(tokenizer_type="mecab_orig", decomposition_type="decomposed_pure")
# wp = WordPieceTokenizer(model_path="./resources/v5_without_dummy_letter/mecab_orig_decomposed_pure_wp-64k/tok.model")
# mecab = MeCabTokenizer_orig(tokenizer_type="mecab_orig", decomposition_type="decomposed_morphological") # ['▁나', '▁ㄴㅡㄴ', '▃', '▁장', '풍', '▁ㅇㅡㄹ', '▃', '▁ㅎㅐㅆ', '▁ㄷㅏ', '▁.']
# wp = WordPieceTokenizer(model_path="./resources/v5_without_dummy_letter/mecab_orig_decomposed_morphological_wp-64k/tok.model")
#
#
# # mecab_fixed.py
# mecab = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="composed", space_symbol= "▃", dummy_letter= "" ) # ['▁나', '▁는', '▃', '▁장풍', '▁을', '▃', '▁하', '▁았', '▁다', '▁.']
# wp = WordPieceTokenizer(model_path="./resources/v5_without_dummy_letter/mecab_fixed_composed_wp-64k/tok.model")
# mecab = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure", space_symbol= "▃", dummy_letter= "" )
# wp = WordPieceTokenizer(model_path="./resources/v5_without_dummy_letter/mecab_fixed_decomposed_pure_wp-64k/tok.model")
# mecab = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="decomposed_morphological", space_symbol= "▃", dummy_letter= "" )
# wp = WordPieceTokenizer(model_path="./resources/v5_without_dummy_letter/mecab_fixed_decomposed_morphological_wp-64k/tok.model")
# mecab = MeCabTokenizer_fixed(tokenizer_type="mecab_fixed", decomposition_type="decomposed_morphological", space_symbol= "▃", dummy_letter= "" )
# wp = WordPieceTokenizer(model_path="./resources/v4_without_dummy_letter/mecab_fixed_decomposed_morphological_wp-32k/tok.model")
#
#
#
# # mecab_all.py
# mecab = MeCabTokenizer_all(token_type="eojeol", tokenizer_type="mecab_fixed", decomposition_type="composed",  grammatical_symbol=["⫸", "⭧"])    # ['▁전태', '희는', '▁한국', '대학교에', '▁묵', '었었다']
# wp = WordPieceTokenizer(model_path="./resources/v6_without_dummy_letter_grammatical_symbol_F/eojeol_mecab_fixed_composed_grammatical_symbol_F_wp-64k/bert_tokenizer.json")

# mecab = MeCabTokenizer_all(token_type="eojeol", tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure", grammatical_symbol=["⫸", "⭧"]) # ['▁전태', '희는', '▁한국', '대학교에', '▁묵', '어', 'ᆻ었다']
# wp = WordPieceTokenizer(model_path="./resources/v6_without_dummy_letter/eojeol_mecab_fixed_decomposed_pure_wp-64k/tok.vocab")
#
# mecab = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="composed")    # ['▁전태', '희', '▁는', '▃', '▁한국', '▁대학교', '▁에', '▃', '▁묵', '▁었었', '▁다']     # ['▁난', '▃', '▁널', '▃', '▁좋아해']
# wp = WordPieceTokenizer(model_path="./resources/v6_without_dummy_letter/morpheme_mecab_orig_composed_wp-64k/tok.vocab")
# mecab = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="decomposed_pure", grammatical_symbol=["⫸", "⭧"]) # ['▁전태', '희', '▁는', '▃', '▁한국', '▁대학교', '▁에', '▃', '▁묵', '▁었었', '▁다']
# wp = WordPieceTokenizer(model_path="./resources/v6_without_dummy_letter/morpheme_mecab_orig_decomposed_pure_wp-64k/tok.vocab")
#     mecab = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="decomposed_morphological", grammatical_symbol=["⫸", "⭧"]) # ['▁전태', '희', '▁는', '▃', '▁한국', '▁대학교', '▁에', '▃', '▁묵', '▁었었', '▁다']
#     wp = WordPieceTokenizer(model_path="./resources/v6_without_dummy_letter/morpheme_mecab_orig_decomposed_morphological_wp-64k/tok.vocab")
#
# mecab = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="composed", space_symbol="", grammatical_symbol=["⫸", "⭧"])    # ['▁전태', '희', '▁는', '▃', '▁한국', '▁대학교', '▁에', '▃', '▁묵', '▁었었', '▁다']    # ['▁나', '▁ᆫ', '▃', '▁너', '▁ᆯ', '▃', '▁좋아하', '▁아']
# wp = WordPieceTokenizer(model_path="./resources/v6_without_dummy_letter/morpheme_mecab_fixed_composed_wp-64k/tok.vocab")
#     mecab = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure", grammatical_symbol=["⫸", "⭧"]) # ['▁전태', '희', '▁는', '▃', '▁한국', '▁대학교', '▁에', '▃', '▁묵', '▁었었', '▁다']
#     wp = WordPieceTokenizer(model_path="./resources/v6_without_dummy_letter/morpheme_mecab_fixed_decomposed_pure_wp-64k/tok.vocab")
#     mecab = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="decomposed_morphological", grammatical_symbol=["⫸", "⭧"]) # ['▁전태', '희', '▁는', '▃', '▁한국', '▁대학교', '▁에', '▃', '▁묵', '▁었었', '▁다']
#     wp = WordPieceTokenizer(model_path="./resources/v6_without_dummy_letter/morpheme_mecab_fixed_decomposed_morphological_wp-64k/tok.vocab")
#
#
#
#
#
#
# mecab = MeCabTokenizer_all(token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="composed", grammatical_symbol="⭧") # ['▁전태', '희', '▁는', '▃', '▁한국', '▁대학교', '▁에', '▃', '▁묵', '▁었었', '▁다']
# wp = WordPieceTokenizer(model_path="./output_wp/morpheme_mecab_orig_composed_wp-0k/tok.model")
#
#
#
#
#
# # mc-wp
# text = "나는 널 좋아해"
# text = "전태희는 널 좋아해"
# mc = MeCabWordPieceTokenizer(mecab=mecab, wp=wp)
# mecab.tokenize(text)
# mc.tokenize(text)
#
#
#
# # text = '나는 오늘 저녁을 먹었다.'   # ['▁나', '▁는', '▃', '▁오늘', '▃', '▁저녁', '▁을', '▃', '▁먹', '▁', '었', '▁다', '▁.']
# # text = "대한민국에 우리끼리 살아보자"    # ['▁대한민국', '▁에', '▃', '▁우리', '▁끼', '리', '▃', '▁살', '▁아', '▁보', '▁자']
# text = "사망 플래그의 좋은 예시이다."
#
# text = "내가 먹다"
# text = "카기"
# text = "텔레비전을 보고 있는데 전화벨이 울렸다"
#
#
#
# text = "나는 장풍을 했다."
# text = "나는 장소를 했다."
#
# text = "전태희는 한국대학교에 묵었었다"
#
#
# text = "사람은 머리는 크다"
# text = "날씨가 춥다"
# text = "뱃사람이 있다"
# text = '사람을 죽인 죄'
# text = '목을 졸라 콜라에게 준 죄'
# text = '널 위해 준 선물'
#
# len(mc.tokenize(text)[0])
# len(mc.tokenize(text)[1])
# len(mc.tokenize(text)[2])
#
#
# self = mc
# self.tokenize(text)
#
#
# def show_tokenized(mecab, wp, text):
#     mc = MeCabWordPieceTokenizer(mecab=mecab, wp=wp)
#     print(mc.tokenize(text))
#
#
# show_tokenized(mecab, wp, text)
#
#
#
# ## v6
# text ="래미안이 좋다"
#
# text = "사망 플래그의 좋은 예시이다."
# # eojeol
#     # '사망 플래그의 좋은 예시이다'
#     # ▁플래: 4970 ▁플래그:36169      # ▁예: 167   _예시: 25430
#     # '플래그의'가 없으니 빈도 높은 '플래' 선택
#
# ['▁사망', '▁플래', '그의', '▁좋은', '▁예', '시이다', '.']
# ['▁사망', '▁플래', '그의', '▁좋은', '▁예', '시이다', '.']
# ['▁사망', '▁플래', '그의', '▁좋은', '▁예', '시이다', '.']
#
# # morpheme orig
#     # '사망 플래그 의 좋 은 예시 이 다 .'
#     # ▁플래: 3708 ▁플래그: 17307     # ▁예: 178   ▁예시: 8714
#     # '플래그'가 있으니 그대로 선택
#
# ['▁사망', '▃', '▁플래그', '▁의', '▃', '▁좋', '▁은', '▃', '▁예시', '▁이', '▁다', '▁.']
# ['▁사망', '▃', '▁플래그', '▁의', '▃', '▁좋', '▁은', '▃', '▁예시', '▁이', '▁다', '▁.']
# ['▁사망', '▃', '▁플래그', '▁의', '▃', '▁좋', '▁은', '▃', '▁예시', '▁이', '▁다', '▁.']
#
#
#
# # morpheme orig morphological
# text = "사람은 머리는 크다"
# ['▁사람', '▁은', '▃', '▁머리', '▁는', '▃', '▁크', '▁다']
#
#
#
# # eojeol morphological      # ▁하았다: 393      # ▁하: 58
# text = "나는 장풍을 했다."
# ['▁나는', '▁장', '풍을', '▁하았다', '.']
#
#
#
#
# ## v4
# # with fixed morphological
# ['▁사람', '▁ㅇㅡㄴ', '▃', '▁머리', '▁ㄴㅡㄴ', '▃', '▁크', '▁ㄷㅏ', '⊸']
