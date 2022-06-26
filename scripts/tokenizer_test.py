# BPE 토큰화 결과 테스트용

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


    # tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_orig", decomposition_type = "composed", grammatical_symbol = ["", ""])
    # orig_composed_F = 'orig_composed_F' + '\t' + get_tokenized_result(tokenizer, string)

    # tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_orig", decomposition_type = "decomposed_pure", grammatical_symbol = ["", ""])
    # orig_pure_F = 'orig_pure_F' + '\t' + get_tokenized_result(tokenizer, string)


    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "composed", grammatical_symbol = ["", ""])
    fixed_composed_F = 'fixed_composed_F' + '\t' + get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "decomposed_pure", grammatical_symbol = ["", ""])
    fixed_pure_F = 'fixed_pure_F' + '\t' + get_tokenized_result(tokenizer, string)

    # tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "decomposed_grammatical", grammatical_symbol = ["", ""])
    # fixed_grammatical_F = 'fixed_grammatical_F' + '\t' + get_tokenized_result(tokenizer, string)

    tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "decomposed_lexical", grammatical_symbol = ["", ""])
    fixed_lexical_F = 'fixed_lexical_F' + '\t' + get_tokenized_result(tokenizer, string)

    # # grammatical symbol T
    # tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "composed", grammatical_symbol = ["⫸", "⭧"])
    # fixed_composed_T = 'fixed_composed_T' + '\t' + get_tokenized_result(tokenizer, string)
    #
    # tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "decomposed_pure", grammatical_symbol = ["⫸", "⭧"])
    # fixed_pure_T = 'fixed_pure_T' + '\t'+ get_tokenized_result(tokenizer, string)
    #
    # tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "decomposed_grammatical", grammatical_symbol = ["⫸", "⭧"])
    # fixed_grammatical_T = 'fixed_grammatical_T' + '\t' + get_tokenized_result(tokenizer, string)
    #
    # tokenizer = get_tokenizer(token_type = "morpheme", tokenizer_type = "mecab_fixed", decomposition_type = "decomposed_lexical", grammatical_symbol = ["⫸", "⭧"])
    # fixed_lexical_T = 'fixed_lexical_T' + '\t' + get_tokenized_result(tokenizer, string)


    # return "\n".join(['\n'+eojeol_composed_F, eojeol_pure_F, orig_composed_F, orig_pure_F, fixed_composed_F, fixed_pure_F, fixed_grammatical_F, fixed_lexical_F, fixed_composed_T, fixed_pure_T,fixed_grammatical_T,fixed_lexical_T])





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
#
#
# sent = "쉶퓀이 묿다 균상종속영양생물, "
#
#
# sent = "건대입구역"; show_tokenizations(string=sent)
# sent = "내가 보건대 너는 크게 될 거다"; show_tokenizations(string=sent)
# # 내가 보건 ##대 너는 크게 될 거 ##다
# # 내가 보건 ##대 너는 크게 될 거다
# # 내 가 보 건대 너 는 크 게 되 ᆯ 거 이 다
# # 내 가 보 건대 너 는 크 게 되 ᆯ 거 이 다
# # 내 가 보 건대 너 는 크 게 되 ᆯ 거 이 다
#
# # 건 ##대 ##입구역
# # 건 ##대 ##입구역
# # 건대 ##입구역
# # 건대 ##입구역
# # 건대 ##입구역
#
#
# sent = "히라노 아야 아니?"; show_tokenizations(string=sent)
# sent = "넌 살아야 한다"; show_tokenizations(string=sent)
# # 히라 ##노 아야 ##를 아니 ##?
# # 히라노 아야 ##를 아니 ##?
# # 히라노 아야 를 아니 ?
# # 히라노 아야 를 아니 ?
# # 히라노 아야 를 아니 ?
# # sent = "넌 살아야 한다"; show_tokenizations(string=sent)
# # 넌 살아야 한다
# # 넌 살아야 한다
# # 너 ᆫ 살 아야 하 ᆫ다
# # 너 ᆫ 살 아야 하 ᆫ다
# # 너 ᆫ 살 아야 하 ᆫ다
#
#
# sent = "소서리스의 스킬"; show_tokenizations(string=sent)
# sent = "용서하소서"; show_tokenizations(string=sent)
# # 소서 ##리스의 스킬
# # 소서 ##리스의 스킬
# # 소서 리스 의 스킬
# # 소서 리스 의 스킬
# # 소서 리스 의 스킬
# # sent = "용서하소서"; show_tokenizations(string=sent)
# # 용서 ##하 ##소 ##서
# # 용서 ##하 ##소서
# # 용서 하 소서
# # 용서 하 소서
# # 용서 하 소서
#
#
# sent = "어서어서 끝내자!"; show_tokenizations(string=sent)
# sent = "먹어서 응원하자!"; show_tokenizations(string=sent)
# # 어서 ##어서 끝내 ##자!
# # 어서 ##어서 끝내 ##자!
# # 어서 ##어 ##서 끝내 자 !
# # 어서 ##어 ##서 끝내 자 !
# # 어서 ##어 ##서 끝내 자 !
# # sent = "먹어서 응원하자!"; show_tokenizations(string=sent)
# # 먹어서 응원 ##하자 ##!
# # 먹어서 응원 ##하자 ##!
# # 먹 어서 응원 하 자 !
# # 먹 어서 응원 하 자 !
# # 먹 어서 응원 하 자 !
#
#
#
#
# sent = "라면카페에 갔어."; show_tokenizations(string=sent)
# sent = "너라면 할 수 있어"; show_tokenizations(string=sent)
# # 해 ##물 ##라면 ##이 먹고 싶어 ##.
# # 해 ##물 ##라면 ##이 먹고 싶어.
# # 해물 라면 이 먹 고 싶 어 .
# # 해물 라면 이 먹 고 싶 어 .
# # 해물 라면 이 먹 고 싶 어 .
# # sent = "너라면 할 수 있어"; show_tokenizations(string=sent)
# # 너 ##라면 할 수 있어
# # 너 ##라면 할 수 있어
# # 너 이 라면 하 ᆯ 수 있 어
# # 너 이 라면 하 ᆯ 수 있 어
# # 너 이 라면 하 ᆯ 수 있 어
#
#
#
#
# sent = "보고타에 갔어."; show_tokenizations(string=sent)
# sent = "나보고 그랬어?"; show_tokenizations(string=sent)
#
# sent = "친구는 우크라이나 사람이다"; show_tokenizations(string=sent)
# sent = "벌써 반이나 끝냈다"; show_tokenizations(string=sent)
# # 친구는 우크라이나 사람이 ##다
# # 친구는 우크라이나 사람이 ##다
# # 친구 는 우크라 이나 사람 이 다
# # 친구 는 우크라 이나 사람 이 다
# # 친구 는 우크라 이나 사람 이 다
# # sent = "벌써 반이나 끝냈다"; show_tokenizations(string=sent)
# # 벌써 반이 ##나 끝 ##냈 ##다
# # 벌써 반이 ##나 끝내 ##ᆻ다
# # 벌써 반 이나 끝내 었 다
# # 벌써 반 이나 끝내 었 다
# # 벌써 반 이나 끝내 었 다
#
# sent = "나라면 해물라면을 먹었을걸."; show_tokenizations(string=sent)
# # 나 이 라면 해물 라면 을 먹 었 을 거 이 야 .
# # 나 이 라면 해물 라면 을 먹 었 을 거 이 야 .
#
# sent = "보고타에 다녀왔다"; show_tokenizations(string=sent)
#
# sent = "소서리스의 스킬"; show_tokenizations(string=sent)
# sent = "노답이다"; show_tokenizations(string=sent)
#
