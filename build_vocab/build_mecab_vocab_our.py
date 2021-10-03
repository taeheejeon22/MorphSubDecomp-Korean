# tokenization_v2_0.py로 먼저 토큰화를 하기 때문에 안 써도 됨.

import argparse
import json
import os
import time
from collections import Counter
from functools import partial
from itertools import chain
from multiprocessing import Pool
from typing import List

import MeCab

# INPUT_CORPUS = "./pretrain_corpus/sample_namuwiki_20200302.txt"
INPUT_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_orig/composed/namuwiki_20200302_tokenized_mecab_orig_composed_sample.txt" # orig / composed
INPUT_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_orig/decomposed_pure/namuwiki_20200302_tokenized_mecab_orig_decomposed_pure_sample.txt" # orig / decomposed_pure
INPUT_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_orig/decomposed_morphological/namuwiki_20200302_tokenized_mecab_orig_decomposed_morphological_sample.txt" # orig / decomposed_morphological

INPUT_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_fixed/composed/namuwiki_20200302_tokenized_mecab_fixed_composed_sample.txt"   # fixed /composed
INPUT_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_fixed/decomposed_pure/namuwiki_20200302_tokenized_mecab_fixed_decomposed_pure_sample.txt" # fixed / decomposed_pure
INPUT_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_fixed/decomposed_morphological/namuwiki_20200302_tokenized_mecab_fixed_decomposed_morphological_sample.txt" # fixed / decomposed_morphological

OUTPUT_DIR = "./resources"


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--space_symbol", type=str, default="▃")

    parser.add_argument("--pad_piece", type=str, default="[PAD]", help="index=0")
    parser.add_argument("--unk_piece", type=str, default="[UNK]", help="index=1")
    parser.add_argument("--bos_piece", type=str, default="[BOS]", help="index=2")
    parser.add_argument("--eos_piece", type=str, default="[EOS]", help="index=3")
    parser.add_argument(
        "--special_symbols",
        type=str,
        default="[CLS],[SEP],[MASK]",
        help="Special tokens. You can pass a comma-separated list of special tokens.",
    )
    parser.add_argument("--n_jobs", type=int, default=20)
    args = vars(parser.parse_args())
    print(args)

    output_dir = os.path.join(OUTPUT_DIR, f"mecab-{args['vocab_size']//1000}k")
    os.makedirs(output_dir, exist_ok=True)

    # save arguments info
    output_info_path = os.path.join(output_dir, "build_info.json")
    with open(output_info_path, "w", encoding="utf-8") as f:
        json.dump(args, f, indent=4)

    # # set tokenizing func
    # tokenize_fn = partial(tokenize, space_symbol=args["space_symbol"])
    #
    # counter = Counter()
    # start_time = time.time()
    # print(f"start tokenization ...")
    # with open(INPUT_CORPUS, "r", encoding="utf-8") as f:
    #     with Pool(args["n_jobs"]) as p:
    #         tokenized = p.map(tokenize_fn, f)
    #         counter.update(chain.from_iterable(tokenized))
    # elapsed_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
    # print(f"complete tokenization for all files. (elapsed time: {elapsed_time})")
    #
    #


    #### read a corpus
    with open(INPUT_CORPUS, "r", encoding="utf-8") as f:
        corpus = f.readlines()
        corpus = [line[:-1] for line in corpus] # remove '\n's

    # split the corpus by token
    tokens = [token for line in  [line.split(" ") for line in corpus] for token in line]

    # special tokens
    special_tokens = [args["pad_piece"], args["unk_piece"], args["bos_piece"], args["eos_piece"]]
    special_tokens.extend(args["special_symbols"].split(","))




############
    counter = Counter(tokens)


#
    # # slice with vocab size
    vocab = counter.most_common(args["vocab_size"] - len(special_tokens))


    # print out-of-vocabulary
    total_freq = sum(counter.values())
    # total_freq = sum([item[1] for item in counter])
    oov_freq = total_freq - sum([v[1] for v in vocab])
    # oov_freq = total_freq - sum(vocab.values())
    print(f"oov: {oov_freq}/{total_freq} ({oov_freq * 100.0 / total_freq:.2f}%)")

    # save mecab vocab
    print("write mecab vocab file...")
    output_vocab_path = os.path.join(output_dir, "tok.vocab")
    with open(output_vocab_path, "w", encoding="utf-8") as f:
        for token in special_tokens:
            f.write(f"{token}\t-1\n")
        for token, freq in vocab:
            f.write(f"{token}\t{freq}\n")

    # mecab config
    print("write mecab config file...")
    output_config_path = os.path.join(output_dir, "tok.json")
    with open(output_config_path, "w", encoding="utf-8") as f:
        json.dump(args, f, indent=4)

    # save fairseq vocab
    print("write fairseq vocab file...")
    with open(os.path.join(output_dir, "fairseq.vocab"), "w") as fout:
        with open(os.path.join(output_dir, "tok.vocab"), "r") as fin:
            start_idx = 4 + len(args["special_symbols"].split(","))  # pad, unk, bos, eos + special_symbols
            for line in fin.readlines()[start_idx:]:
                splitted = line.split("\t")
                fout.write(f"{' '.join(splitted)}")

    print("done.")