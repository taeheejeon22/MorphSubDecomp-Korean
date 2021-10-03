from typing import List

from tokenizer.base import BaseTokenizer
# from tokenizer.mecab import MeCabTokenizer
from tokenizer.mecab import MeCabFixedTokenizer

from tokenizer.sentencepiece import SentencePieceTokenizer




class MeCabFixedSentencePieceTokenizer(BaseTokenizer):
    def __init__(self, mecab: MeCabFixedTokenizer, sp: SentencePieceTokenizer):
        self.mecab = mecab
        self.mecab = MeCabFixedTokenizer("./resources/mecab_fixed_sp-0k/tok.json")
        self.sp = sp

    def tokenize(self, text: str) -> List[str]:
        tokenized = self.mecab.tokenize(text)   # tokenized = ['카터', '는', '▃', '미국', '이다']

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




mc = MeCabFixedSentencePieceTokenizer(mecab="./resources/mecab_fixed_sp-0k/tok.json", sp="./resources/mecab_fixed_sp-0k/tok.model")




mc.tokenize('나는 오늘 저녁을 먹었다.')

self = mc

text = "너를 죽이겠다"