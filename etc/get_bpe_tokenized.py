##
import argparse
import os
import time

from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

import sys
sys.path.insert(0, '.')

from tokenizer.get_tokenizer import get_tokenizer


# load a corpus
def load_corpus(corpus_path: str):
    with open(corpus_path, "r") as f:
        corpus = f.readlines()
    corpus = [line[:-1] for line in corpus]
    return corpus




# Tokenizer = get_tokenizer(tokenizer_name=tokenizer_name, resource_dir=resource_dir,
#                           token_type=token_type,
#                           tokenizer_type=tokenizer_type,
#                           decomposition_type=decomposition_type,
#                           space_symbol=space_symbol,
#                           dummy_letter=dummy_letter, nfd=nfd,
#                           grammatical_symbol=grammatical_symbol,
#                           skip_special_tokens=skip_sepcial_tokens)

# Tokenizer.tokenize("훌륭한 예시이다")



# tokenize_fun("훌륭한 예시이다")
# tokenize_fun("훏✏뷁(ㄴㅇㄹ")

# fn = partial(tokenize_fun)
#
# threads = 16
#
# with Pool(threads) as p:
#     tokenized_corpus = p.map(fn, corpus)    # 라인별로 mecab + BPE 토큰화한 코퍼스



def save_tokenized_corpus(args, tokenized_corpus: list):
    os.makedirs("./corpus/bpe_tokenized", exist_ok=True)

    output_path = os.path.join("./corpus/bpe_tokenized", "_".join([args["token_type"], args["tokenizer_type"], args["decomposition_type"], "grammatical_symbol", "F" if args["grammatical_symbol"] == ["", ""] else "T", "txt"]) )

    # os.makedirs(output_dir,  exist_ok=True)

    with open(output_path, "w") as f:
        for ix in range(len(tokenized_corpus)):
            f.write(" ".join(tokenized_corpus[ix]) + "\n")





# def main(args):
#     corpus = load_corpus(args["corpus_path"])
#
#     Tokenizer = get_tokenizer(tokenizer_name=args["tokenizer_name"], resource_dir=args["resource_dir"],
#                               token_type=args["token_type"],
#                               tokenizer_type=args["tokenizer_type"],
#                               decomposition_type=args["decomposition_type"],
#                               space_symbol=args["space_symbol"],
#                               dummy_letter=args["dummy_letter"], nfd=args["nfd"],
#                               grammatical_symbol=args["grammatical_symbol"],
#                               skip_special_tokens=args["skip_sepcial_tokens"])
#
#     example = Tokenizer.tokenize("훌륭한 예시이다")
#
#     def tokenize_fun(text: str):
#         tokenized = Tokenizer.tokenize(text)
#         return tokenized
#
#     print(f"tokenized exmaple: {example}")
#
#     fn = partial(tokenize_fun)
#
#
#     with Pool(args["threads"]) as p:
#         tokenized_corpus = p.map(fn, corpus)    # 라인별로 mecab + BPE 토큰화한 코퍼스
#
#
#     save_tokenized_corpus(args=args, tokenized_corpus=tokenized_corpus)






# unk_cnt = [line.count("[UNK]") for line in tokenized_corpus]
# sum(unk_cnt)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # args = {"resource_dir": "./resources/v6_without_dummy_letter_grammatical_symbol_F",
    # "tokenizer_name": "morpheme_mecab_orig_composed_grammatical_symbol_F_wp-64k",
    #
    # "token_type": "morpheme",
    # "tokenizer_type": "mecab_fixed",
    # "decomposition_type": "composed",
    #
    # "space_symbol": "",
    # "dummy_letter": "",
    # "nfd": True,
    # "grammatical_symbol": ["", ""],
    # "skip_sepcial_tokens": False}


    parser.add_argument("--corpus_path", type=str, default="/home/jht/rsync/namuwiki_20210301_with_preprocessing_v5_kss.txt")
    parser.add_argument("--tokenizer_name", type=str, default="")
    parser.add_argument("--resource_dir", type=str, default="")

    parser.add_argument("--token_type", type=str, default="")   # eojeol / morpheme # v2에서 추가
    parser.add_argument("--tokenizer_type", type=str, default="mecab_orig")  # mecab_orig / mecab_fixed
    parser.add_argument("--decomposition_type", type=str, default="composed")   # "composed", "decomposed_pure", "decomposed_morphological"
    parser.add_argument("--space_symbol", type=str, default="")  # "▃" chr(9603)
    parser.add_argument("--dummy_letter", type=str, default="")  # 초성/중성/종성 자리 채우기용 더미 문자. default는 없음(""). # "⊸"  # chr(8888)
    parser.add_argument("--grammatical_symbol", type=list, default=["", ""])  # ["⫸", "⭧"] # chr(11000) # chr(11111)
    parser.add_argument("--nfd", action="store_true", default=False)   # NFD 사용해서 자모 분해할지
    parser.add_argument("--skip_sepcial_tokens", action="store_true", default=False)   # BPE 특수 문자 출력할지

    parser.add_argument("--threads", type=int, default=16)

    args = vars(parser.parse_args())
    print(args)



    start_time = time.time()
    print(f"\n\nstart ...")



    ### main
    corpus = load_corpus(args["corpus_path"])

    Tokenizer = get_tokenizer(tokenizer_name=args["tokenizer_name"], resource_dir=args["resource_dir"],
                              token_type=args["token_type"],
                              tokenizer_type=args["tokenizer_type"],
                              decomposition_type=args["decomposition_type"],
                              space_symbol=args["space_symbol"],
                              dummy_letter=args["dummy_letter"], nfd=args["nfd"],
                              grammatical_symbol=args["grammatical_symbol"],
                              skip_special_tokens=args["skip_sepcial_tokens"])

    example = Tokenizer.tokenize("훌륭한 예시이다")

    def tokenize_fun(text: str):
        tokenized = Tokenizer.tokenize(text)
        return tokenized

    print(f"tokenized exmaple: {example}")

    fn = partial(tokenize_fun)


    with Pool(args["threads"]) as p:
        tokenized_corpus = p.map(fn, corpus)    # 라인별로 mecab + BPE 토큰화한 코퍼스


    save_tokenized_corpus(args=args, tokenized_corpus=tokenized_corpus)
    ###





    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"complete tokenization for all files. (elapsed time: {elapsed_time})\n")




# python etc/get_bpe_tokenized.py --corpus_path="/home/jth/rsync/namuwiki_20210301_with_preprocessing_v5_kss.txt" \
#                             --tokenizer_name="eojeol_mecab_fixed_composed_grammatical_symbol_F_wp-64k" \
#                             --resource_dir="./resources/v6_without_dummy_letter_grammatical_symbol_F" \
#                             --token_type="eojeol" --tokenizer_type="mecab_fixed" --decomposition_type="composed" \
#                             --nfd --threads=58
#                              # --grammatical_symbol=⫸⭧