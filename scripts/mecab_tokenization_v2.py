# v2
# kortok 토크나이저 버리고 내 거로. tokenizers_acl_v3.py


# 토크나이저 mecab_fixed 이용해서 만들기
# none은 그냥 전처리 끝난 거 복붙하면 됨
# 만들어진 tok.json은 수동으로 resources로 옮기기

import argparse
import json
import os
import re
import time
from functools import partial
from multiprocessing import Pool
from typing import List


import sys
sys.path.insert(0, '.')

from tokenizer.old.mecab_fixed import str2jamo, mecab_tokenize
# from tokenizer.mecab_fixed import str2jamo as str2jamo


# from tokenizer.mecab_fixed_v2 import MeCabTokenizer_fixed
# import scripts.tokenizers_acl_v2 as tok
import scripts.tokenizers_acl_v3 as tok


import MeCab

# INPUT_CORPUS = "./dataset/wiki/sample_ko-wiki-200420.txt"
# INPUT_CORPUS = "../wikiko_20210901_with_preprocessing_v2.txt"
# OUTPUT_DIR = "./dataset/wiki/mecab_tokenized"

TOKENIZER = MeCab.Tagger(f"--dicdir /usr/local/lib/mecab/dic/mecab-ko-dic")

grammatical_pos = ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC", "EP", "EF", "EC", "ETN", "ETM"]    # 어미, 조사


# kortok API based (tokenizer/mecab_orig.py)
def tokenize_kortok(text: str, tokenizer_type: str, decomposition_type: str, space_symbol: str = "▃", dummy_letter: str = "") -> List[str]:
    assert (tokenizer_type in ["mecab_orig", "mecab_fixed"] ), 'check the tokenizer type!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
    assert (decomposition_type in ["composed", "decomposed_pure", "decomposed_morphological"] ), 'check the decomposition type!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'


    text = text.strip()
    text_ptr = 0
    tokenized = []
    for mor in TOKENIZER.parse(text).split("\n"):
        if "\t" in mor:
            splitted = mor.split("\t")
            token = splitted[0]
            pos = splitted[1].split(",", 1)[0]

            if text[text_ptr] == " ":
                while text[text_ptr] == " ":
                    text_ptr += 1
                assert text[text_ptr] == token[0]

                tokenized.append(space_symbol)

            # tokenized.append(token)

            if tokenizer_type == "mecab_orig":   # mecab original
                if decomposition_type == "composed":
                    tokenized.append(token)
                elif decomposition_type == "decomposed_pure":
                    tokenized.append(str2jamo(token, grammatical=False, dummy_letter=dummy_letter))   # 자모 분해 후 추가
                elif decomposition_type == "decomposed_morphological":
                    if sum([1 for pos in pos.split("+") if pos in grammatical_pos]) < 1:  # VV+EC 등 고려해도 문법 형태소 없으면
                        tokenized.append(token) # 그대로 추가
                    elif sum([1 for pos in pos.split("+") if pos in grammatical_pos]) >= 1:  # VV+EC 등 고려해서 문법 형태소 있으면
                        tokenized.append(str2jamo(token, grammatical=False, dummy_letter=dummy_letter))   # 자모 분해 후 추가

            # elif tokenizer_type == "mecab_fixed":    # mecab fixed
            #     if decomposition_type == "composed":
            #         mecab_tokenized = [mor_pos[0] for mor_pos in mecab_tokenize(mor)]  # ['나', 'ᆫ'] 진짜 형태소로 쪼개진 토큰들 저장
            #         tokenized += mecab_tokenized
            #     elif decomposition_type == "decomposed_pure":
            #         mecab_tokenized = [mor_pos[0] for mor_pos in mecab_tokenize(mor)]  # ['나', 'ᆫ'] 진짜 형태소로 쪼개진 토큰들 저장
            #         tokenized += [str2jamo(token, grammatical=False, dummy_letter=dummy_letter) for token in mecab_tokenized] # 자모 분해 후 추가
            #     elif decomposition_type == "decomposed_morphological":
            #         mecab_tokenized_with_pos = mecab_tokenize(mor)[:]  # [('나', 'NP'), ('ᆫ', 'JX')] 진짜 형태소로 쪼개진 토큰들 저장 with POS tag
            #         tokenized += [mor_pos[0] if (not mor_pos[-1] in grammatical_pos) else str2jamo(mor_pos[0], grammatical=False, dummy_letter=dummy_letter) for mor_pos in mecab_tokenized_with_pos]    # 어휘 형태소는 그대로, 문법 형태소는 자모 분해 후 추가

            text_ptr += len(token)

    return tokenized


# our (konlpy based)
def tokenize_our(text: str, token_type: str, tokenizer_type: str, decomposition_type: str, space_symbol: str = "▃", dummy_letter: str = "") -> List[str]:
    text = text.strip()

    # return self.tok.mecab_tokenizer(text, use_original=self.use_original, pure_decomposition=self.pure_decomposition, morphological=self.morphological)
    return tok.mecab_tokenizer(text, token_type=token_type, tokenizer_type=tokenizer_type, decomposition_type=decomposition_type)





# tokenize("안녕? 뭐 해!", use_original=False, decomposition_type="composed")
#
#
# text = "내셔날 보그"
# text = "엘림에듀 등이 있다"
# text = "인터내셔널 그룹"
# text = "인터내셔날 그룹"
# text = "캘리포니아"
# text = "내셔널지오그래픽"
# tokenize(text)
#
#
# from scripts._mecab import Mecab
# mc = Mecab()
# mc.morphs(text)
# mc.pos(text, flatten=False)


if __name__ == "__main__":
    # # wiki ko
    # corpus = "wikiko_20210901"
    # # INPUT_CORPUS = "../wikiko_20210901_with_preprocessing_v2.txt"
    # INPUT_CORPUS = "../wikiko_20210901_with_preprocessing_v3_nn.txt"
    #
    # # # namuwiki
    # # corpus = "namuwiki_20200302"
    # # INPUT_CORPUS = "../namuwiki_20200302_with_preprocessing_v3_nn.txt"


    # OUTPUT_DIR = "../tokenized/"


    parser = argparse.ArgumentParser()
    parser.add_argument("--space_symbol", type=str, default="▃")
    parser.add_argument("--n_jobs", type=int, default=16)

        # 추가한 것들
    # parser.add_argument("--use_original", type=bool, default=True)  # mecab orig / fixed
    parser.add_argument("--token_type", type=str, default="")   # eojeol / morpheme # v2에서 추가
    parser.add_argument("--tokenizer_type", type=str, default="mecab_orig")  # mecab_orig / mecab_fixed

    parser.add_argument("--decomposition_type", type=str, default="composed")   # "composed", "decomposed_pure", "decomposed_morphological"
    parser.add_argument("--dummy_letter", type=str, default="") # 초성/중성/종성 자리 채우기용 더미 문자. default는 없음("").

    # parser.add_argument("--corpus", type=str)   # "wikiko", "namuwiki"
    parser.add_argument("--corpus_path", type=str)



    # args = {"space_symbol": "▃", "n_jobs": 16, "use_original": True, "decomposition_type": "composed", "dummy_letter": ""}
    # args = {"space_symbol": "▃", "n_jobs": 16, "use_original": False, "decomposition_type": "decomposed_morphological", "dummy_letter": ""}


    args = vars(parser.parse_args())
    print(args)


    # 출력 디렉토리 생성
    if args["dummy_letter"] == "":  # 초성/중성/종성 자리 채우기용 더미 문자 안 쓰면
        OUTPUT_DIR = "./corpus/tokenized/" + "without_dummy_letter/"
    else:
        OUTPUT_DIR = "./corpus/tokenized/" + "with_dummy_letter/"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # set tokenizing func
    # tokenize_fn = partial(tokenize, space_symbol=args["space_symbol"])
        # mc = MeCabTokenizer_fixed(use_original=args["use_original"], decomposition_type=args["decomposition_type"], space_symbol=args["space_symbol"], dummy_letter=args["dummy_letter"])
        # tokenize_fn = partial(mc.tokenize)


    # if args["corpus"] == "wikiko":
    #     # wiki ko
    #     corpus = "wikiko_20210901"
    #     # INPUT_CORPUS = "../wikiko_20210901_with_preprocessing_v2.txt"
    #     INPUT_CORPUS = "../wikiko_20210901_with_preprocessing_v3_nn.txt"
    # elif args["corpus"] == "namuwiki":
    #     # namuwiki
    #     corpus = "namuwiki_20200302"
    #     INPUT_CORPUS = "../namuwiki_20200302_with_preprocessing_v3_nn.txt"


    INPUT_CORPUS = args["corpus_path"]
    corpus = "_".join(INPUT_CORPUS.split("/")[-1].split("_")[:2])

    p_endswith_num = re.compile("\d+$") # split 커맨드로 나눈 후 생기는 번호 검색용

    if p_endswith_num.search(INPUT_CORPUS): # 숫자로 끝나면. 즉 split 커맨드로 나뉜 파일이라면
        part_num = p_endswith_num.search(INPUT_CORPUS).group()
        file_name = "_".join([corpus, args["token_type"], args["tokenizer_type"], args["decomposition_type"], part_num]) + ".txt"
    else:
        file_name = "_".join([corpus, args["token_type"], args["tokenizer_type"], args["decomposition_type"]]) + ".txt"




    # if "orig" in args["tokenizer_type"]:
    #     tokenize_fn = partial(tokenize_kortok, tokenizer_type=args["tokenizer_type"], decomposition_type=args["decomposition_type"], space_symbol=args["space_symbol"], dummy_letter=args["dummy_letter"] )
    # elif "fixed" in args["tokenizer_type"]:
    #     tok = tok.tokenizers(dummy_letter=args["dummy_letter"], space_symbol=args["space_symbol"])
    #     tokenize_fn = partial(tokenize_our, tokenizer_type=args["tokenizer_type"], decomposition_type=args["decomposition_type"], space_symbol=args["space_symbol"], dummy_letter=args["dummy_letter"] )


    # v2
    tok = tok.tokenizers(dummy_letter=args["dummy_letter"], space_symbol=args["space_symbol"])
    tokenize_fn = partial(tokenize_our, token_type=args["token_type"], tokenizer_type=args["tokenizer_type"], decomposition_type=args["decomposition_type"], space_symbol=args["space_symbol"], dummy_letter=args["dummy_letter"] )




    # tokenization
    # str_corpus = args["corpus"]
    str_tokenizer_type = args["tokenizer_type"]
    str_decomposition_type = args["decomposition_type"]
    print(f"corpus: {INPUT_CORPUS}\n")
    print(f"tokenizer_type: {str_tokenizer_type}\n")
    print(f"deocmposition_type: {str_decomposition_type}\n")


    start_time = time.time()
    print(f"start tokenization ...")

    # if args["tokenizer_type"] == "none":    # 형태소 분석하지 않고 원문 그대로 이용
    #     with open(INPUT_CORPUS, "r", encoding="utf-8") as f:
    #         tokenized = f.read()
    #
    # else:   # 형태소 분석할 경우
    #     with open(INPUT_CORPUS, "r", encoding="utf-8") as f:
    #         with Pool(args["n_jobs"]) as p:
    #             tokenized = p.map(tokenize_fn, f)

    # v2
    with open(INPUT_CORPUS, "r", encoding="utf-8") as f:
        with Pool(args["n_jobs"]) as p:
            tokenized = p.map(tokenize_fn, f)




    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"complete tokenization for all files. (elapsed time: {elapsed_time})")

    # mecab tokenized corpus
    # if args["use_original"] == True:
    #     tokenizer_type = "mecab_orig"
    # elif args["use_original"] == False:
    #     tokenizer_type = "mecab_fixed"
    #

    # set a input path automatically





    OUTPUT_DIR_sub = OUTPUT_DIR + "_".join([corpus, args["token_type"], args["tokenizer_type"] ]) + "/" + args["decomposition_type"]

    os.makedirs(OUTPUT_DIR_sub, exist_ok=True)


    # 저장
    if args["tokenizer_type"] == "none":  # 형태소 분석하지 않고 원문 그대로 이용
        with open(os.path.join(OUTPUT_DIR_sub, os.path.basename(file_name)), "w", encoding="utf-8") as f:
            f.write(tokenized)

    else:  # 형태소 분석할 경우
        with open(os.path.join(OUTPUT_DIR_sub, os.path.basename(file_name)), "w", encoding="utf-8") as f:
            for tokens in tokenized:
                f.write(" ".join(tokens) + "\n")

    # mecab config
    print("write mecab config file...")
    output_config_path = os.path.join(OUTPUT_DIR_sub, "tok.json")
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(args, f, indent=4)


    print("saved in:", os.path.join(OUTPUT_DIR_sub, os.path.basename(file_name)) )
    print("done.")
