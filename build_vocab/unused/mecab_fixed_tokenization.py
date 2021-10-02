import argparse
import json
import os
import time
from functools import partial
from multiprocessing import Pool
from typing import List
from itertools import chain

from konlpy.tag import Mecab
# import MeCab

# INPUT_CORPUS = "./pretrain_corpus/namuwiki/sample_namuwiki_20200302.txt"
INPUT_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_orig/composed/namuwiki_20200302_tokenized_mecab_orig_composed.txt" # orig / composed
INPUT_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_orig/decomposed_pure/namuwiki_20200302_tokenized_mecab_orig_decomposed_pure_.txt" # orig / decomposed_pure
INPUT_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_orig/decomposed_morphological/namuwiki_20200302_tokenized_mecab_orig_decomposed_morphological.txt" # orig / decomposed_morphological

INPUT_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_fixed/composed/namuwiki_20200302_tokenized_mecab_fixed_composed.txt"   # fixed /composed
INPUT_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_fixed/decomposed_pure/namuwiki_20200302_tokenized_mecab_fixed_decomposed_pure.txt" # fixed / decomposed_pure
INPUT_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_fixed/decomposed_morphological/namuwiki_20200302_tokenized_mecab_fixed_decomposed_morphological.txt" # fixed / decomposed_morphological


OUTPUT_DIR = "./dataset/wiki/mecab_tokenized_fixed"

# TOKENIZER = MeCab.Tagger(f"--dicdir /usr/local/lib/mecab/dic/mecab-ko-dic")


# def tokenize(text: str, space_symbol: str = "▃") -> List[str]:
#     text = text.strip()
#     text_ptr = 0
#     tokenized = []
#     for mor in TOKENIZER.parse(text).split("\n"):
#         if "\t" in mor:
#             splitted = mor.split("\t")
#             token = splitted[0]
#             # pos = splitted[1].split(",", 1)[0]
#
#             if text[text_ptr] == " ":
#                 while text[text_ptr] == " ":
#                     text_ptr += 1
#                 assert text[text_ptr] == token[0]
#
#                 tokenized.append(space_symbol)
#
#             tokenized.append(token)
#             text_ptr += len(token)
#
#     return tokenized


# for inserting space_symbol ("▃")
# https://stackoverflow.com/questions/5920643/add-an-item-between-each-item-already-in-the-list
def intersperse(lst, item):
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result

mc = Mecab(use_original=False)

def tokenize(text, space_symbol):   # "▃"
    mor_poss = mc.pos(text, flatten=False)  # [[('이것', 'NP'), ('이', 'JKC')], [('아니', 'VCN'), ('다', 'EC')]]
    mors = [[mor_pos[0] for mor_pos in word] for word in mor_poss] # [['이것', '이'], ['아니', '다']]
    return list (chain.from_iterable( intersperse(mors, space_symbol) ) )    # ['이것', '이', '▃', '아니', '다']



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--space_symbol", type=str, default="▃")
    parser.add_argument("--n_jobs", type=int, default=20)
    args = vars(parser.parse_args())
    print(args)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # set tokenizing func
    tokenize_fn = partial(tokenize, space_symbol=args["space_symbol"])

    start_time = time.time()
    print(f"start tokenization ...")
    with open(INPUT_CORPUS, "r", encoding="utf-8") as f:
        with Pool(args["n_jobs"]) as p:
            tokenized = p.map(tokenize_fn, f)
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    print(f"complete tokenization for all files. (elapsed time: {elapsed_time})")

    # mecab tokenized corpus
    with open(os.path.join(OUTPUT_DIR, os.path.basename(INPUT_CORPUS)), "w", encoding="utf-8") as f:
        for tokens in tokenized:
            f.write(" ".join(tokens) + "\n")

    # mecab config
    print("write mecab config file...")
    output_config_path = os.path.join(OUTPUT_DIR, "tok.json")
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(args, f, indent=4)

    print("done.")
