from typing import List

import sentencepiece as spm

from tokenizer.base import BaseTokenizer


class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self, model_path: str, reverse: bool = False):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

        # self.sp.Load("./resources/v3_without_dummy_letter/32k/tok.model")

        self.reverse = reverse

    def tokenize(self, text: str) -> List[str]:
        if self.reverse:
            tokenized = self.sp.EncodeAsPieces(text[::-1].strip())
            tokenized = [s[::-1] for s in tokenized][::-1]
        else:
            tokenized = self.sp.EncodeAsPieces(text.strip())
            # sp.encode(text.strip())

        return tokenized

    def detokenize(self, tokens: List[str]) -> str:
        text = "".join(tokens).replace("▁", " ").strip()
        return text



# model_path="./output_sp/morpheme_mecab_orig_composed_sp-0k/tok.model"
# model_path="./output_sp/morpheme_mecab_orig_composed_sp-64k_0/tok.model"
#
# sp = spm.SentencePieceProcessor(model_file=model_path)
# sp.Encode("우리 ⭧는 좋 ⭧다", out_type=str)
#
# sp.encode("마을이 넓다", out_type)
# # model_path = "./resources/v3_without_dummy_letter/sp-32k/tok.model"
# sp = SentencePieceTokenizer(model_path)
# text = "대한민국에 우리끼리 살아보자"    # ['▁대한민국에', '▁우리', '끼리', '▁살아', '보자']
# text = "난 널 좋아해"  # ['▁난', '▁널', '▁좋아', '해']
# text = "밥을 먹습니다"  # ['▁밥', '을', '▁먹', '습니다']
#
# text = "우리 ⭧는 좋 ⭧다"
#
# text = '내 ⭧가 먹 ⭧다'
#
# ['내', '⭧가', '먹', '⭧다']
#
# sp.tokenize(text)
#
#
# self = sp