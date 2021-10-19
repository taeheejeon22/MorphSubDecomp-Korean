from typing import List

from tokenizer.base import BaseTokenizer
from tokenizer.mecab_kortok import MeCabTokenizer_kortok
from tokenizer.sentencepiece import SentencePieceTokenizer


class MeCabSentencePieceTokenizer_kortok(BaseTokenizer):
    def __init__(self, mecab: MeCabTokenizer_kortok, sp: SentencePieceTokenizer):
        self.mecab = mecab
        self.sp = sp

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


# mc = MeCabSentencePieceTokenizer(mecab="./resources/mecab-2k/tok.json",sp="./resources/mecab-2k/tok.model")
#
# mc.tokenize(['나는 오늘 저녁을 먹었다.'])