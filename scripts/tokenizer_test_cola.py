import json
import os
import unicodedata
import sys
import pandas as pd

sys.path.insert(0, '.')


from tokenizer import (
    # CharTokenizer,
    # JamoTokenizer,
    MeCabSentencePieceTokenizer_orig,
    MeCabSentencePieceTokenizer_fixed,
    MeCabSentencePieceTokenizer,
    MeCabWordPieceTokenizer,
    # MeCabTokenizer,
    MeCabTokenizer_orig,
    MeCabTokenizer_fixed,
    MeCabTokenizer_all,
    # MeCabSentencePieceTokenizer_kortok,
    # MeCabTokenizer_kortok,
    SentencePieceTokenizer,
    WordPieceTokenizer,
    Vocab,
    # WordTokenizer,
)



def get_tokenizer(token_type: str, tokenizer_type: str, decomposition_type: str, space_symbol: str = "", dummy_letter: str = "", grammatical_symbol: list = ["", ""]):
    use_grammatical_symbol = "grammatical_symbol_F" if grammatical_symbol == ["", ""] else "grammatical_symbol_T"

    wp_path = "./resources/v6_without_dummy_letter_" + use_grammatical_symbol
    sub_path = "_".join([token_type, tokenizer_type, decomposition_type, use_grammatical_symbol, "wp-64k"]) + "/bert_tokenizer.json"

    model_path = wp_path + "/" + sub_path
    # print(f"\nmodel path: {model_path}\n")

    wp = WordPieceTokenizer(model_path)

    if (token_type == "eojeol") and (decomposition_type == "composed"):
        tokenizer = wp

    else:
        mecab = MeCabTokenizer_all(token_type=token_type, tokenizer_type=tokenizer_type,
                           decomposition_type=decomposition_type,
                           space_symbol=space_symbol, dummy_letter=dummy_letter,
                           nfd=True, grammatical_symbol=grammatical_symbol)

        tokenizer = MeCabWordPieceTokenizer(mecab=mecab, wp=wp)  # mecab_wp.py

    return tokenizer


def get_tokenized_result(tokenizer, string, nfd: bool = True):
    # if nfd == True:
    #     string = str_to_nfd(string)

    tokenized = tokenizer.tokenize(string)
    print(" ".join(tokenized))
    
    return " ".join(tokenized)

    # nfd 확인용
    # for token in tokenized:
    #     print(f"len: {len(token)}")




# token_type = "eojeol"; tokenizer_type = "mecab_fixed"; decomposition_type = "composed"; grammatical_symbol = False
# tokenizer = get_tokenizer(token_type = "eojeol", tokenizer_type = "mecab_fixed", decomposition_type = "composed", grammatical_symbol = False)
#
# get_tokenized_result(tokenizer, "안녕하세요")




def show_tokenizations(string):
    # grammatical symbol F
    tokenizer = get_tokenizer(token_type = "eojeol", tokenizer_type = "mecab_fixed", decomposition_type = "composed", grammatical_symbol = ["", ""])
    eojeol_composed_F = 'eojeol_composed_F' + '\t' + get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "eojeol", tokenizer_type = "mecab_fixed", decomposition_type = "decomposed_pure", grammatical_symbol = ["", ""])
    eojeol_pure_F = 'eojeol_pure_F' + '\t' + get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_orig", decomposition_type = "composed", grammatical_symbol = ["", ""])
    orig_composed_F = 'orig_composed_F' + '\t' + get_tokenized_result(tokenizer, string)
    

    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_orig", decomposition_type = "decomposed_pure", grammatical_symbol = ["", ""])
    orig_pure_F = 'orig_pure_F' + '\t' + get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "composed", grammatical_symbol = ["", ""])
    fixed_composed_F = 'fixed_composed_F' + '\t' + get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "decomposed_pure", grammatical_symbol = ["", ""])
    fixed_pure_F = 'fixed_pure_F' + '\t' + get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "decomposed_grammatical", grammatical_symbol = ["", ""])
    fixed_grammatical_F = 'fixed_grammatical_F' + '\t' + get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "decomposed_lexical", grammatical_symbol = ["", ""])
    fixed_lexical_F = 'fixed_lexical_F' + '\t' + get_tokenized_result(tokenizer, string)

    # grammatical symbol T
    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "composed", grammatical_symbol = ["⫸", "⭧"])
    fixed_composed_T = 'fixed_composed_T' + '\t' + get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "decomposed_pure", grammatical_symbol = ["⫸", "⭧"])
    fixed_pure_T = 'fixed_pure_T' + '\t'+ get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "decomposed_grammatical", grammatical_symbol = ["⫸", "⭧"])
    fixed_grammatical_T = 'fixed_grammatical_T' + '\t' + get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "decomposed_lexical", grammatical_symbol = ["⫸", "⭧"])
    fixed_lexical_T = 'fixed_lexical_T' + '\t' + get_tokenized_result(tokenizer, string)

    return "\n".join(['\n'+eojeol_composed_F, eojeol_pure_F, orig_composed_F, orig_pure_F, fixed_composed_F, fixed_pure_F, fixed_grammatical_F, fixed_lexical_F, fixed_composed_T, fixed_pure_T,fixed_grammatical_T,fixed_lexical_T])  






# sent = "뱌뵵뵤벼벼벼추퓨를 먹었다."
# sent = "소고기덮밥을 먹었다"
# sent = "담임선생님은 나를 좋아해"
# sent = "프랑스령 북아프리카를 좋아해"
# sent = "나랑 쇼핑하자"
# sent = "나는 너를 좋아해"
# sent = "난 선생님이 너무너무 좋아"
# sent = "난 이 옷이 너무너무 좋은데 진짜루 비쌌어"

# sent = "이 친구가 그럴 사람이 아닌데 실수를 했었나 봐"
# sent = "난 탈락할 줄 알았  이 친구는 그럴 사람이 아닌데."
# sent = "이 친구는 그럴 사람이 아닌데 너무너무 많이 샀어"
# sent = "그렇긴 해도 난 이 옷이 너무너무 지쳤어"
# sent = '난 그럴지도 몰라'
# sent = "난 이 부분이 그럴지도 몰라"
# sent = "난 왜 그런지도 몰라"
# sent = "이 지도는 축척이 그럴지도 몰라"
# sent = "그럴지도 모르지만 이 옷이 비쌌어" # 맞다
# sent = "이 옷이 비쌀지도 모르지만 난 샀어"
# sent = "이 옷이 비쌀지도 모르지만 난 구매했어"
# sent = "이옷이 비싼데 난 샀는데"
# show_tokenizations(string=sent)




### cola
# 전처리

from typing import Dict, List, Tuple

def load_data(file_path: str) -> Tuple[List[str]]:
    """
    file_path에 존재하는 tsv를 읽어서 bert_data.InputIds 형태로 변경해주는 함수입니다.
    각각의 row를 bert input으로 바꾸어주기 위한 함수입니다.
    각 row는 아래처럼 구성되어야 합니다.
    1. sentence
    3. label
    """
    sentences: List[str] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()):
            splitted = line.strip().split("\t")
            if len(splitted) != 2:
                #print(f"[ERROR] {repr(line)}, line {i}")
                continue
            sentences.append(splitted[0])
            # labels.append(label_to_index[splitted[1]])

    return sentences


cola_test = load_data('dataset/nlu_tasks/cola/cola_test.tsv')

with open('tokenized_result/cola_test.csv', "w", encoding='utf-8') as f:
    show_tokenizations("뱌뵵뵤벼벼벼추퓨를 먹었다.")
    for sent1 in (cola_test):
        f.write(show_tokenizations(string=sent1))
