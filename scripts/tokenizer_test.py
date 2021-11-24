import json
import os
import unicodedata
import sys

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
    print(f"\nmodel path: {model_path}\n")

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
    get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "eojeol", tokenizer_type = "mecab_fixed", decomposition_type = "decomposed_pure", grammatical_symbol = ["", ""])
    get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_orig", decomposition_type = "composed", grammatical_symbol = ["", ""])
    get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_orig", decomposition_type = "decomposed_pure", grammatical_symbol = ["", ""])
    get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "composed", grammatical_symbol = ["", ""])
    get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "decomposed_pure", grammatical_symbol = ["", ""])
    get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "decomposed_grammatical", grammatical_symbol = ["", ""])
    get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "decomposed_lexical", grammatical_symbol = ["", ""])
    get_tokenized_result(tokenizer, string)

    # grammatical symbol T
    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "composed", grammatical_symbol = ["⫸", "⭧"])
    get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "decomposed_pure", grammatical_symbol = ["⫸", "⭧"])
    get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "decomposed_grammatical", grammatical_symbol = ["⫸", "⭧"])
    get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "decomposed_lexical", grammatical_symbol = ["⫸", "⭧"])
    get_tokenized_result(tokenizer, string)





sent = "소고기덮밥을 먹었다"
sent = "담임선생님은 나를 좋아해"
sent = "프랑스령 북아프리카를 좋아해"
sent = "나랑 쇼핑하자"
sent = "나는 너를 좋아해"
sent = "난 선생님이 너무너무 좋아"
sent = "난 이 옷이 너무너무 좋은데 진짜루 비쌌어"

sent = "이 친구가 그럴 사람이 아닌데 실수를 했었나 봐"
sent = "난 탈락할 줄 알았  이 친구는 그럴 사람이 아닌데."
sent = "이 친구는 그럴 사람이 아닌데 너무너무 많이 샀어"
sent = "그렇긴 해도 난 이 옷이 너무너무 지쳤어"
sent = '난 그럴지도 몰라'
sent = "난 이 부분이 그럴지도 몰라"
sent = "난 왜 그런지도 몰라"
sent = "이 지도는 축척이 그럴지도 몰라"
sent = "그럴지도 모르지만 이 옷이 비쌌어" # 맞다
sent = "이 옷이 비쌀지도 모르지만 난 샀어"
sent = "이 옷이 비쌀지도 모르지만 난 구매했어"
show_tokenizations(string=sent)