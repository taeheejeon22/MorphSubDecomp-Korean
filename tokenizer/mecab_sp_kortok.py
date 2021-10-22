from typing import List

from tokenizer.base import BaseTokenizer
from tokenizer.mecab_kortok import MeCabTokenizer_kortok
from tokenizer.sentencepiece import SentencePieceTokenizer


class MeCabSentencePieceTokenizer_kortok(BaseTokenizer):
    def __init__(self, mecab: MeCabTokenizer_kortok, sp: SentencePieceTokenizer):
        self.mecab = mecab
        self.sp = sp

        # self.mecab = MeCabTokenizer_kortok(config_path="./resources/mecab_orig_composed_sp-64k/tok.json")
        # self.sp = SentencePieceTokenizer(model_path="./resources/mecab_orig_composed_sp-64k/tok.model")


    def tokenize(self, text: str) -> List[str]:
        tokenized = self.mecab.tokenize(text)
        tokenized = self.sp.tokenize(" ".join(tokenized))

        output = []
        for i in range(0, len(tokenized)):
            if i + 1 < len(tokenized) and (tokenized[i] == "▁" and tokenized[i + 1] == "▃"):
                continue
            if tokenized[i] == "▁▃":
                tokenized[i] = "▃"
            output.append(tokenized[i])

        return output

    def detokenize(self, tokens: List[str]) -> str:
        text = "".join(tokens).replace("▁", "").replace(" ", "").replace("▃", " ").strip()
        return text


# mc = MeCabSentencePieceTokenizer_kortok(mecab=MeCabTokenizer_kortok, sp=SentencePieceTokenizer)
# self = mc
# # text = '나는 오늘 저녁을 먹었다.'   # ['▁나', '▁는', '▃', '▁오늘', '▃', '▁저녁', '▁을', '▃', '▁먹', '▁', '었', '▁다', '▁.']
# # text = "대한민국에 우리끼리 살아보자"    # ['▁대한민국', '▁에', '▃', '▁우리', '▁끼', '리', '▃', '▁살', '▁아', '▁보', '▁자']
# # text = "사망 플래그의 좋은 예시이다."
# # text = "나는 장풍을 했다."
# text = "난 널 좋아해"    # ['▁난', '▃', '▁널', '▃', '▁좋아해']
# mc.tokenize(text)
