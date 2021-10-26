# v4: tokenizers_acl_v2 연동

# v2_1: 문서 간 띄어쓰기 2회 하기

# tokenizers_acl 이용하는 버전 (개인 논문도 이 방식으로 교체해야)

import gzip
import os

import numpy as np
import pickle
import re
from tqdm import tqdm
# import jamo_functions_v1 as jamo    # mecab orig 사용 시 '잔다' VV+EC 등을 자모 분해하지 않음 '잔다'
# import jamo_functions_v2 as jamo    # mecab orig 사용 시 '잔다' VV+EC 등을 자모 분해함  'ㅈㅏㄴㄷㅏ'
# import jamo_functions_v3 as jamo
                                    # corpus v3
                                    # 자음만 있는 조사, 어미 종성에 위치하도록
                                    # 나 --ㄴ 고양이 이 ㄷㅏ-
                                    # '준 사람' > '주 --ㄴ 사람'
from scripts.tokenizers_acl import tokenizers



# load corpus
def load_corpus(corpus_path):
    # with gzip.open(corpus_path, "rb") as f:
    #     corpus = pickle.load(f)
    #     sent_lst = corpus.split("\n")   # 문장 단위로 분절

    # del f, corpus

    with open(corpus_path, "r") as f:
        sent_lst = f.readlines()

    sent_lst = [sent[:-1] if sent != "\n" else sent for sent in sent_lst]   # remove "\n"s created by readlines()



    print(len(sent_lst))    # 62,481,588
    # 46,540,361 46,350,425 46,241,968 45,242,757  45,070,098  44,846,837  44,715,408  38,698,423  38,942,110  38,756,655

    return sent_lst







# def tokenization(sent_lst, token_type, composition_type, use_original):
def tokenization(sent_lst, analyzer, composition_type, use_original):
    tok = tokenizers(dummy_letter=dummy_letter, space_symbol=space_symbol)  # tokenizer instance

    if analyzer == "none":  # 형태소 분석하지 않고 어절 그대로 쓴다면
        # tokenized_corpus = [tok.eojeol_tokenizer(sent) for sent in tqdm(sent_lst, position=0, leave=True)]

        return sent_lst

    elif "mecab" in analyzer:

        if composition_type == "composed":  # mecab + 음절 수준 (kakao)
            tokenized_corpus = [tok.mecab_tokenizer(sent, use_original=use_original, pure_decomposition=False, morphological=False) for sent in tqdm(sent_lst, position=0, leave=True)]

            # mc = Mecab(use_original=use_original)
            # # tokenized_corpus = [mc.morphs(sent_lst[ix]) for ix in tqdm( range(len(sent_lst)), position=0, leave=True )]
            # tokenized_corpus = [mc.morphs(sent) for sent in tqdm(sent_lst, position=0, leave=True)]

        elif composition_type == "decomposed_pure":    # mecab + pure decomposition
            tokenized_corpus = [tok.mecab_tokenizer(sent, use_original=use_original, pure_decomposition=True, morphological=False) for sent in tqdm(sent_lst, position=0, leave=True)]

            # mc = Mecab(use_original=use_original)
            # # tokenized_corpus = [jamo.str2jamo( " ".join(mc.morphs(sent_lst[ix]) ) ).split(" ") for ix in tqdm(range(len(sent_lst)), position=0, leave=True)]
            # tokenized_corpus = [jamo.str2jamo(" ".join(mc.morphs(sent))).split(" ") for sent in tqdm(sent_lst, position=0, leave=True)]

        elif composition_type == "decomposed_morphological": # mecab + morphological decomposition
            # tokenized_corpus = [tok.mecab_with_morphological_decomposition(sent, use_original=use_original) for sent in tqdm(sent_lst, position=0, leave=True)]
            tokenized_corpus = [tok.mecab_tokenizer(sent, use_original=use_original, pure_decomposition=False, morphological=True) for sent in tqdm(sent_lst, position=0, leave=True)]

            # tokenized_corpus = [jamo.str2jamo_morphological(sent, morpheme_analysis=True, use_original=use_original).split(" ") for sent in tqdm(sent_lst, position=0, leave=True)]

        return tokenized_corpus


# tokenized_composed[:2]
# tokenized_decomposed_pure[:2]
# tokenized_decomposed_morphological[:2]


# tokenized_corpus = [mc.morphs(sent_lst[ix]) for ix in tqdm( range(len(sent_lst)), position=0, leave=True )]
#
# tcs = list()
# for sent in tqdm(sent_lst, position=0, leave=True ):
#     tcs.append( mc.morphs(sent))




# make directory
def make_directory(analyzer, composition_type):
    # save_dir = "./pretrain_corpus/tokenized/" + "namuwiki_" + analyzer + "/" + composition_type
    save_dir = "../tokenized/" + corpus_name + "_" + analyzer + "/" + composition_type
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True) # a Python version of $mkdir -p


# save as a txt
def save_decomposed_corpus(file_name, analyzer, composition_type, corpus):
    # save_path = "./pretrain_corpus/tokenized/" + "namuwiki_" + "/".join([analyzer, composition_type])
    save_path = "../tokenized/" + corpus_name + "_" + analyzer + "/" + composition_type
    file_path = save_path + "/" + file_name

    with open(file_path, "w") as f:
        for ix in range(len(corpus)):
            if analyzer == "none":
                if corpus[ix] != "\n":  # 문서 사이 공백 아니면
                    f.write("".join(corpus[ix]) + "\n")
                elif corpus[ix] == "\n":  # 문서 사이 공백이면
                    f.write("".join(corpus[ix]))
            else:
                if corpus[ix] != "\n":  # 문서 사이 공백 아니면
                    f.write(" ".join(corpus[ix]) + "\n")
                elif corpus[ix] == "\n":  # 문서 사이 공백이면
                    f.write(" ".join(corpus[ix]))

    print(f"saved in {file_path}")



def main(sent_lst, analyzer, composition_type, use_original):
    print(f"\nanalyzer: {analyzer}\n")
    print(f"\ncomposition type: {composition_type}\n")
    print(f"\nuse original: {use_original}\n")

    # if analyzer == "none":
    #     # tokenization
    #     tokenized_corpus = tokenization(sent_lst=sent_lst, analyzer=analyzer, composition_type=composition_type, use_original=use_original)
    #
    #     # save
    #     make_directory(analyzer=analyzer, composition_type=composition_type)
    #
    #     # save
    #     if use_original == True:
    #         mecab_type = "mecab_orig"
    #     elif use_original == False:
    #         mecab_type = "mecab_fixed"
    #
    #     file_name = "namuwiki_20200302_tokenized_" + "_".join([analyzer, composition_type, mecab_type,]) + ".txt"
    #     save_decomposed_corpus(file_name=file_name, analyzer=analyzer, composition_type=composition_type, corpus=tokenized_corpus)
    #
    #
    # else:
    # memory 문제로 나눠서 처리
    # iter = 5  # all


    if "namu" in corpus_name:
        iter = 3
    elif "ko" in corpus_name:
        iter = 1

    # for ix in range(iter):
    for ix in range(2, iter):
    # for ix in range(0, 1):
        print(f"\niteration: {ix}\n")
        begin_idx = 10000000*ix
        end_idx = 10000000 * (ix + 1)

        # if token_type == "eojeol":
        #     morph_analysis = False
        # elif "morpheme" in token_type:
        #     morph_analysis = True

        if ix < iter-1:
            tokenized_corpus = tokenization(sent_lst=sent_lst[begin_idx:end_idx], analyzer=analyzer, composition_type=composition_type, use_original=use_original)

        elif ix == iter-1:
            tokenized_corpus = tokenization(sent_lst=sent_lst[begin_idx:], analyzer=analyzer, composition_type=composition_type, use_original=use_original)

        # make a directory to save the result
        make_directory(analyzer=analyzer, composition_type=composition_type)

        # save
        # if use_original == True:
        #     mecab_type = "mecab_orig"
        # elif use_original == False:
        #     mecab_type = "mecab_fixed"

        if "namu" in corpus_name:
            file_name = corpus_name + "_tokenized_" + "_".join([analyzer, composition_type, str(ix)]) + ".txt"
        elif "ko" in corpus_name:
            file_name = corpus_name + "_tokenized_" + "_".join([analyzer, composition_type]) + ".txt"

        save_decomposed_corpus(file_name=file_name, analyzer=analyzer, composition_type=composition_type, corpus=tokenized_corpus)

        del tokenized_corpus



if __name__ == "__main__":
    corpus_path = "../namuwiki_20200302_with_preprocessing_v3_nn.txt" # namuwiki
    corpus_name = "namuwiki_20200302"

    # corpus_path = "../wikiko_20210901_with_preprocessing_v2.txt"  # wikiko
    # corpus_name = "wikiko_20210901"
    # corpus_path = "../wikiko_20210901_with_preprocessing_v3_nn.txt"  # wikiko
    # corpus_name = "wikiko_20211021"

    sent_lst = load_corpus(corpus_path=corpus_path) # 62,481,588
    print(sent_lst[58:63])  # 문서 사이 공백 확인



    # sent_lst = sent_lst[:1000]
    #
    # with open("./pretrain_corpus/sample_ko-wiki-200420.txt", "r") as f:
    #     sent_lst = f.readlines()
    #
    # sent_lst = [sent[:-1] for sent in sent_lst]


    # # corpuse size check
    # len_sent = [len(txt.splitlines()) for txt in sent_lst]
    #
    # len_ej =  [len(txt.split(" ")) for txt in sent_lst]
    #
    # sum(len_sent)   # 62,481,588
    # sum(len_ej) # 607,037,254



    # dummy_letter = "⊸"  # chr(8888)
    dummy_letter = ""  # none
    space_symbol = "▃"  # chr(9603)


    # p_kakao = re.compile(r"[^가-힣\x20-\x7F]*")  # 타 언어 문자, 특수 기호 제거
    # # p_kakao = re.compile(r"[^a-z]")
    # sent = 'ab⊸⊸c▃가d事'
    # p_kakao.search(sent)
    # p_kakao.sub("", sent)
    # # re.sub(p_kakao, sent, "e")
    # re.sub(p_kakao, "ㄸ", sent)
    #
    # p_asc = re.compile("[^\x20-\x7F]*")
    # sent = 'abc가'
    # p_asc.search(sent)
    # re.sub(p_asc, sent, "")



    ## 0. baseline: eojeol
    analyzer = "none"
    use_original = True
    composition_type = "composed"
    main(sent_lst=sent_lst, analyzer=analyzer, composition_type=composition_type, use_original=use_original)

    ## 1. mecab original
    analyzer = "mecab_orig"
    use_original = True
    # 1) composed # ['난', '▃', '널', '▃', '좋', '아', '해']
    composition_type = "composed"
    main(sent_lst=sent_lst, analyzer=analyzer, composition_type=composition_type, use_original=use_original)

    # 2) decomposed pure # ['ㄴㅏㄴ', '▃', 'ㄴㅓㄹ', '▃', 'ㅈㅗㅎ', 'ㅇㅏ', 'ㅎㅐ']
    composition_type = "decomposed_pure"
    main(sent_lst=sent_lst, analyzer=analyzer, composition_type=composition_type, use_original=use_original)

    # 3) decomposed morphological # ['ㄴㅏㄴ', '▃', 'ㄴㅓㄹ', '▃', '좋', 'ㅇㅏ', 'ㅎㅐ']
    composition_type = "decomposed_morphological"
    main(sent_lst=sent_lst, analyzer=analyzer, composition_type=composition_type, use_original=use_original)


    ## 2. mecab fixed
    analyzer = "mecab_fixed"
    use_original = False
    # 1) composed # ['나', 'ㄴ', '▃', '너', 'ㄹ', '▃', '좋아하', '아']
    composition_type = "composed"
    main(sent_lst=sent_lst, analyzer=analyzer, composition_type=composition_type, use_original=use_original)

    # 2) decomposed pure # ['ㄴㅏ', 'ㄴ', '▃', 'ㄴㅓ', 'ㄹ', '▃', 'ㅈㅗㅎㅇㅏㅎㅏ', 'ㅇㅏ']
    composition_type = "decomposed_pure"
    main(sent_lst=sent_lst, analyzer=analyzer, composition_type=composition_type, use_original=use_original)

    # 3) decomposed morphological # ['나', 'ㄴ', '▃', '너', 'ㄹ', '▃', '좋아하', 'ㅇㅏ']
    composition_type = "decomposed_morphological"
    main(sent_lst=sent_lst, analyzer=analyzer, composition_type=composition_type, use_original=use_original)
