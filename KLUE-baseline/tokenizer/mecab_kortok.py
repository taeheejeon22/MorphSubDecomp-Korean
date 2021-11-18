
import json
import os
from typing import List

import MeCab

from tokenizer.base import BaseTokenizer


class MeCabTokenizer_kortok(BaseTokenizer):
    def __init__(self, config_path: str):
        self.mecab = MeCab.Tagger(f"--dicdir /usr/local/lib/mecab/dic/mecab-ko-dic")
        with open(config_path) as f:
            self.config: dict = json.load(f)

    def tokenize(self, text: str) -> List[str]:
        text = text.strip()
        text_ptr = 0
        tokenized = []
        for mor in self.mecab.parse(text).split("\n"):
            if "\t" in mor:
                splitted = mor.split("\t")
                token = splitted[0]
                # pos = splitted[1].split(",", 1)[0]

                if text[text_ptr] == " ":
                    while text[text_ptr] == " ":
                        text_ptr += 1
                    assert (
                            text[text_ptr] == token[0]
                    ), f"{repr(text)}//{text_ptr}//{text[text_ptr]}//{token}//{token[0]}\n"

                    tokenized.append(self.config["space_symbol"])

                tokenized.append(token)
                text_ptr += len(token)

        return tokenized

    def detokenize(self, tokens: List[str]) -> str:
        text = "".join(tokens).replace("▃", " ").strip()
        return text



# config_path = "./resources/v2_with_dummy_letter/wikiko_all_64k/mecab_fixed_decomposed_morphological_sp-64k/tok.json"
# mc = MeCabTokenizer_kortok(config_path)
# #
# mc.tokenize("사람은 너를 원해.\n아파르트헤이트는 큰 문제였다.\n너를 죽이겠다.")
# mc.tokenize("사람은 너를 원해.\n아파르트헤이트는 큰 문제였다.\n")
#
# text = "사람은 너를 원해.\n"
# text = "사람은 너를 원해.\n아파르트헤이트는 큰 문제였다.\n"
# text = "사람은 너를 원해.\n너를 죽이겠다.\n"
#
# mc.tokenize(text)   # ['사람', '은', '너', '를', '원해', '.']
