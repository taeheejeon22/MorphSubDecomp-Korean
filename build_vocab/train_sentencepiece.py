import argparse
import json
import os

import sentencepiece as spm

# namuwiki
# INPUT_KO_CORPUS = "./pretrain_corpus/tokenized/namuwiki_none/composed/namuwiki_20200302_tokenized_none_composed_mecab_orig.txt"
# wiki ko
INPUT_KO_CORPUS = "../tokenized/wikiko_20210901_none/composed//home/user/Desktop/git/acl_tokenization/tokenized/wikiko_20210901_none/composed/wikiko_20210901_tokenized_none_composed.txt.txt"

INPUT_EN_CORPUS = "./dataset/wiki/sample_en-wiki-200420.txt"  # for English SentencePiece(BPE) Tokenizer
# INPUT_MECAB_TOKENIZED_CORPUS = "./dataset/wiki/mecab_tokenized/sample_ko-wiki-200420.txt"  # for MeCab-SentencePiece Tokenizer
# INPUT_MECAB_TOKENIZED_CORPUS = "./dataset/wiki/mecab_tokenized_fixed/sample_ko-wiki-200420.txt"  # for MeCab-SentencePiece Tokenizer
# INPUT_MECAB_TOKENIZED_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_orig/composed/namuwiki_20200302_tokenized_mecab_orig_composed.txt" # orig / composed
# INPUT_MECAB_TOKENIZED_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_orig/decomposed_pure/namuwiki_20200302_tokenized_mecab_orig_decomposed_pure.txt" # orig / decomposed_pure
# INPUT_MECAB_TOKENIZED_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_orig/decomposed_morphological/namuwiki_20200302_tokenized_mecab_orig_decomposed_morphological.txt" # orig / decomposed_morphological
#
# INPUT_MECAB_TOKENIZED_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_fixed/composed/namuwiki_20200302_tokenized_mecab_fixed_composed.txt"   # fixed /composed
# INPUT_MECAB_TOKENIZED_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_fixed/decomposed_pure/namuwiki_20200302_tokenized_mecab_fixed_decomposed_pure.txt" # fixed / decomposed_pure
# INPUT_MECAB_TOKENIZED_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_fixed/decomposed_morphological/namuwiki_20200302_tokenized_mecab_fixed_decomposed_morphological.txt" # fixed / decomposed_morphological



# OUTPUT_DIR = "./resources"
OUTPUT_DIR = "./output_sp"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--character_coverage", type=str, default=1.0)
    parser.add_argument(
        "--normalization_rule_name",
        type=str,
        default="identity",
        choices=["nmt_nfkc", "nfkc", "nmt_nfkc_cf", "nfkc_cf", "identity"],
    )  # set "nmt_nfkc" for english training
    parser.add_argument("--pad_piece", type=str, default="[PAD]", help="index=0")
    parser.add_argument("--unk_piece", type=str, default="[UNK]", help="index=1")
    parser.add_argument("--bos_piece", type=str, default="[BOS]", help="index=2")
    parser.add_argument("--eos_piece", type=str, default="[EOS]", help="index=3")
    parser.add_argument("--unk_surface", type=str, default="[UNK]")
    parser.add_argument(
        "--special_symbols",
        type=str,
        default="[CLS],[SEP],[MASK]",
        help="Special tokens. You can pass a comma-separated list of special tokens.",
    )
    parser.add_argument(
        "--tokenizer_type", type=str, default="none", choices=["ko", "en", "none", "mecab_orig", "mecab_fixed"]
    )  # ko: Korean Wiki Corpus, en: English Wiki Corpus, mecab_orig: NamuWiki Corpus tokenized by MeCab_orig, mecab_fixed: NamuWiki Corpus tokenized by MeCab_fixed
    parser.add_argument(
        "--composition_type", type=str, default="composed", choices=["composed", "decomposed_pure", "decomposed_morphological"]
    )  # composed: syllable-level   decomposed_pure: jamo-level     decomposed_morphological: syllable+jamo-level

    args = vars(parser.parse_args())
    print(args)

    tokenizer_type = args["tokenizer_type"]
    composition_type = args["composition_type"]

    # set a input path automatically
    # corpus = "namuwiki_20200302"  # namuwiki
    corpus = "wikiko_20210901"  # wiki ko

    if "mecab" in args["tokenizer_type"]:
        INPUT_MECAB_TOKENIZED_CORPUS = f"../tokenized/{corpus}_{tokenizer_type}/{composition_type}/{corpus}_tokenized_{tokenizer_type}_{composition_type}.txt"  # all
        # INPUT_MECAB_TOKENIZED_CORPUS = f"./pretrain_corpus/tokenized/namuwiki_{tokenizer_type}/{composition_type}/namuwiki_20200302_tokenized_{tokenizer_type}_{composition_type}_half.txt" # half

        # namuwiki
        # INPUT_MECAB_TOKENIZED_CORPUS = f"../tokenized/namuwiki_{tokenizer_type}/{composition_type}/namuwiki_20200302_tokenized_{tokenizer_type}_{composition_type}_all.txt"  # all
        # # INPUT_MECAB_TOKENIZED_CORPUS = f"./pretrain_corpus/tokenized/namuwiki_{tokenizer_type}/{composition_type}/namuwiki_20200302_tokenized_{tokenizer_type}_{composition_type}_half.txt" # half

        # INPUT_MECAB_TOKENIZED_CORPUS = "./pretrain_corpus/tokenized/namuwik_" + args[tokenizer_]  mecab_orig/composed/namuwiki_20200302_tokenized_mecab_orig_composed.txt"  # orig / composed
        # INPUT_MECAB_TOKENIZED_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_orig/decomposed_pure/namuwiki_20200302_tokenized_mecab_orig_decomposed_pure.txt"  # orig / decomposed_pure
        # INPUT_MECAB_TOKENIZED_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_orig/decomposed_morphological/namuwiki_20200302_tokenized_mecab_orig_decomposed_morphological.txt"  # orig / decomposed_morphological
        #
        # INPUT_MECAB_TOKENIZED_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_fixed/composed/namuwiki_20200302_tokenized_mecab_fixed_composed.txt"  # fixed /composed
        # INPUT_MECAB_TOKENIZED_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_fixed/decomposed_pure/namuwiki_20200302_tokenized_mecab_fixed_decomposed_pure.txt"  # fixed / decomposed_pure
        # INPUT_MECAB_TOKENIZED_CORPUS = "./pretrain_corpus/tokenized/namuwiki_mecab_fixed/decomposed_morphological/namuwiki_20200302_tokenized_mecab_fixed_decomposed_morphological.txt"  # fixed / decomposed_morphological
    else:
        INPUT_MECAB_TOKENIZED_CORPUS = f"../tokenized/{corpus}_{tokenizer_type}/{composition_type}/{corpus}_tokenized_{tokenizer_type}_{composition_type}.txt"  # all
        # INPUT_MECAB_TOKENIZED_CORPUS = f"./pretrain_corpus/tokenized/namuwiki_{tokenizer_type}/{composition_type}/namuwiki_20200302_tokenized_{tokenizer_type}_{composition_type}_half.txt" # half



    # set output dir
    if args["tokenizer_type"] == "none":
        input_corpus = INPUT_MECAB_TOKENIZED_CORPUS
        output_dir = os.path.join(OUTPUT_DIR, f"sp-{int(args['vocab_size']) // 1000}k")
    elif args["tokenizer_type"] == "ko":
        input_corpus = INPUT_KO_CORPUS
        output_dir = os.path.join(OUTPUT_DIR, f"sp-{int(args['vocab_size'])//1000}k")
    elif args["tokenizer_type"] == "en":
        input_corpus = INPUT_EN_CORPUS
        output_dir = os.path.join(OUTPUT_DIR, f"en_sp-{int(args['vocab_size'])//1000}k")
    elif args["tokenizer_type"] == "mecab_tokenized":
        input_corpus = INPUT_MECAB_TOKENIZED_CORPUS
        output_dir = os.path.join(OUTPUT_DIR, f"mecab_sp-{int(args['vocab_size'])//1000}k")
    elif "mecab" in args["tokenizer_type"]:
        input_corpus = INPUT_MECAB_TOKENIZED_CORPUS
        output_dir = os.path.join(OUTPUT_DIR, f"{tokenizer_type}_{composition_type}_sp-{int(args['vocab_size'])//1000}k")

    else:
        raise ValueError



    os.makedirs(output_dir, exist_ok=True)

    # save arguments info
    output_info_path = os.path.join(output_dir, "build_info.json")
    with open(output_info_path, "w", encoding="utf-8") as f:
        json.dump(args, f, indent=4)

    cmd = f"--input={input_corpus} "
    cmd += f"--model_prefix={os.path.join(output_dir, 'tok')} "
    cmd += f"--vocab_size={args['vocab_size']} "
    cmd += f"--model_type=bpe "
    cmd += f"--character_coverage={args['character_coverage']} "
    cmd += f"--normalization_rule_name={args['normalization_rule_name']} "
    cmd += f"--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
    cmd += f"--pad_piece={args['pad_piece']} "
    cmd += f"--unk_piece={args['unk_piece']} "
    cmd += f"--bos_piece={args['bos_piece']} "
    cmd += f"--eos_piece={args['eos_piece']} "
    cmd += f"--unk_surface={args['unk_surface']} "
    cmd += f"--user_defined_symbols={args['special_symbols']} "

    # train sentencepiece
    spm.SentencePieceTrainer.Train(cmd)

    # fairseq vocab
    with open(os.path.join(output_dir, "fairseq.vocab"), "w") as fout:
        with open(os.path.join(output_dir, "tok.vocab"), "r") as fin:
            start_idx = 4 + len(args["special_symbols"].split(","))  # pad, unk, bos, eos + special_symbols
            for line in fin.readlines()[start_idx:]:
                splitted = line.split("\t")
                fout.write(f"{' '.join(splitted)}")



    # mecab config
    tok_json = dict()
    tok_json["dummy_letter"] = "⊸"
    tok_json["space_symbol"] = "▃"
    # tok_json["n_jobs"] =
    if "orig" in tokenizer_type:
        tok_json["use_original"] = True
    elif "fixed" in tokenizer_type:
        tok_json["use_original"] = False

    if composition_type == "composed":
        tok_json["pure_decomposition"] = False
        tok_json["morphological"] = False

    elif composition_type == "decomposed_pure":
        tok_json["pure_decomposition"] = True
        tok_json["morphological"] = False

    elif composition_type == "decomposed_morphological":
        tok_json["pure_decomposition"] = False
        tok_json["morphological"] = True
    else:
        tok_json["pure_decomposition"] = None
        tok_json["morphological"] = None


    with open(os.path.join(output_dir, "tok.json"), "w") as f:
        json.dump(tok_json, f, indent=4)
