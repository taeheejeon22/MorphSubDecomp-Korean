# https://huggingface.co/docs/tokenizers/python/latest/pipeline.html#all-together-a-bert-tokenizer-from-scratch

# 다음 경로 수정해서 with / without 폴더 잘 설정해야.
# INPUT_MECAB_TOKENIZED_CORPUS = f"./corpus/tokenized/with_dummy_letter/{corpus}_{tokenizer_type}/{composition_type}/{corpus}_{tokenizer_type}_{composition_type}.txt"  # all


# resources 만드는 코드
# kortok과 달리 tok.json 여기서 만들도록 수정

import argparse
import json
import os

# from tokenizers import BertWordPieceTokenizer



from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import StripAccents, NFD, NFC

from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing

from tokenizers.trainers import WordPieceTrainer




# # train a wordpiece model   # ver 3.0
# def train_wp(vocab_size: int, files: list, save_path: str):
#     tokenizer = BertWordPieceTokenizer(
#         vocab_file=None,
#         clean_text=True,
#         handle_chinese_chars=True,
#         strip_accents=False,  # Must be False if cased model
#         lowercase=False,
#         wordpieces_prefix="##"
#     )
#
#     tokenizer.train(
#         files=files,
#         limit_alphabet=6000,
#         vocab_size=vocab_size
#     )
#
#
#
#     # save
#     tokenizer.save(os.path.join(save_path, "tok.vocab"))
#
#     return tokenizer


# ver 4.12
def train_wp(vocab_size: int, files: list, save_path: str):
    # set a tokenizer
    bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    bert_tokenizer.normalizer = normalizers.Sequence([StripAccents()])  # normalizer
    bert_tokenizer.pre_tokenizer = WhitespaceSplit()  # pretokenizer

    bert_tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    # train
    trainer = WordPieceTrainer(
        vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    bert_tokenizer.train(files, trainer)


    # save
    # bert_tokenizer.save(os.path.join(save_path, "tok.vocab"))
    bert_tokenizer.save(os.path.join(save_path, "tok.model"))


    return bert_tokenizer







def listdir_fullpath(d):
    return sorted([os.path.join(d, f) for f in os.listdir(d)])



# # BERT 학습용 vocab.txt 만들기
# def save_bert_vocab(output_dir: str):
#     f = open(os.path.join(output_dir, "vocab.txt"), "w", encoding="utf-8")
#     with open(os.path.join(output_dir, "tok.model")) as json_file:
#         json_data = json.load(json_file)
#         for item in json_data["model"]["vocab"].keys():
#             f.write(item + '\n')
#
#     f.close()



# vocab.txt tok.vocab 만들기
def save_wp_vocab(output_dir: str):
    # load tok.model (wp model)
    with open(os.path.join(output_dir, "tok.model"), "r", encoding="utf-8") as f:
        wp_model = json.load(f)

    tok_vocab = wp_model["model"]["vocab"]

    # tok.vocab
    with open(os.path.join(output_dir, "tok.vocab"), "w", encoding="utf-8") as f:
        for item, idx in tok_vocab.items():
            f.write(f"{item}\t{idx}\n")


    # vocab.txt
    with open(os.path.join(output_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        for item in tok_vocab.keys():
            f.write(item + '\n')





if __name__ == "__main__":
    # set a input path automatically
    # corpus = "namuwiki_20200302"  # namuwiki

    # corpus = "wikiko_20210901"  # wiki ko
    # corpus = "wikiko_20211021"  # wiki ko

    OUTPUT_DIR = "./output_sp"


    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, required=True)
    # parser.add_argument("--character_coverage", type=str, default=1.0)
    # parser.add_argument(
    #     "--normalization_rule_name",
    #     type=str,
    #     default="identity",
    #     choices=["nmt_nfkc", "nfkc", "nmt_nfkc_cf", "nfkc_cf", "identity"],
    # )  # set "nmt_nfkc" for english training
    # parser.add_argument("--pad_piece", type=str, default="[PAD]", help="index=0")
    # parser.add_argument("--unk_piece", type=str, default="[UNK]", help="index=1")
    # parser.add_argument("--bos_piece", type=str, default="[BOS]", help="index=2")
    # parser.add_argument("--eos_piece", type=str, default="[EOS]", help="index=3")
    # parser.add_argument("--unk_surface", type=str, default="[UNK]")
    # parser.add_argument(
    #     "--special_symbols",
    #     type=str,
    #     default="[CLS],[SEP],[MASK]",
    #     help="Special tokens. You can pass a comma-separated list of special tokens.",
    # )


    # parser.add_argument("--token_type", type=str, default="")   # eojeol / morpheme
    # parser.add_argument(
    #     "--tokenizer_type", type=str, default="", choices=[ "none", "mecab_orig", "mecab_fixed"]
    # )  # mecab_orig: NamuWiki Corpus tokenized by MeCab_orig, mecab_fixed: NamuWiki Corpus tokenized by MeCab_fixed
    # parser.add_argument(
    #     "--composition_type", type=str, default="composed", choices=["composed", "decomposed_pure", "decomposed_morphological", "decomposed_lexical", "decomposed_grammatical"]
    # )  # composed: syllable-level   decomposed_pure: jamo-level     decomposed_morphological: syllable+jamo-level

    # parser.add_argument("--with_dummy_letter", type=bool, default=False)    # 자모 더미 문자 사용 여부: True, False


    parser.add_argument("--tokenized_corpus_path", type=str, default="")  # 토큰화한 코퍼스 경로


    args = {"vocab_size": 64000,
            "tokenizer_type": "mecab_fixed",
            "composition_type": "composed",
            "token_type": "eojeol",
            "with_dummy_letter": False,
            "tokenized_corpus_path": "./corpus/tokenized/space_F_dummy_F_grammatical_F/eojeol_mecab_fixed/composed",
            # "tokenized_corpus_path": "./convert_bert",
            }


    args = vars(parser.parse_args())
    print(args)



    # tokenized file info.
    with open(os.path.join(args["tokenized_corpus_path"], "tok.json")) as f:
        tok_json = json.load(f)



    # token_type = args["token_type"]
    # tokenizer_type = args["tokenizer_type"]
    # composition_type = args["composition_type"]
    token_type = tok_json["token_type"]
    tokenizer_type = tok_json["tokenizer_type"]
    decomposition_type = tok_json["decomposition_type"]

    grammatical_symbol = "F" if tok_json["grammatical_symbol"] == ["",""] else "T"



    print(token_type, "###############")
    print(tokenizer_type, '#################')


    # # 자모 더미 문자 사용 여부에 따른 경로 설정
    # if args["with_dummy_letter"] == False:
    #     with_dummy_letter = "without_dummy_letter"
    # elif args["with_dummy_letter"] == True:
    #     with_dummy_letter = "with_dummy_letter"





    # set output dir
    input_file_paths = [path for path in listdir_fullpath(args["tokenized_corpus_path"]) if path.endswith(".txt")]

    # listdir_fullpath("./corpus/tokenized/space_F_dummy_F_grammatical_T/wikiko_20210901_morpheme_mecab_orig/composed")
    # input_corpus = INPUT_MECAB_TOKENIZED_CORPUS


    print(f"\n\nINPUT_CORPORA: {input_file_paths}\n\n")


    output_dir = os.path.join(OUTPUT_DIR, f"{token_type}_{tokenizer_type}_{decomposition_type}_grammatical_symbol_{grammatical_symbol}_wp-{int(args['vocab_size']) // 1000}k")

    # if args["tokenizer_type"] == "none":
    #     input_corpus = INPUT_MECAB_TOKENIZED_CORPUS
    #     output_dir = os.path.join(OUTPUT_DIR, f"sp-{int(args['vocab_size']) // 1000}k")
    # # elif args["tokenizer_type"] == "ko":
    # #     input_corpus = INPUT_KO_CORPUS
    # #     output_dir = os.path.join(OUTPUT_DIR, f"sp-{int(args['vocab_size'])//1000}k")
    # # elif args["tokenizer_type"] == "en":
    # #     input_corpus = INPUT_EN_CORPUS
    # #     output_dir = os.path.join(OUTPUT_DIR, f"en_sp-{int(args['vocab_size'])//1000}k")
    #
    # # elif "mecab" in args["tokenizer_type"]:
    # #     input_corpus = INPUT_MECAB_TOKENIZED_CORPUS
    # #     output_dir = os.path.join(OUTPUT_DIR, f"mecab_sp-{int(args['vocab_size'])//1000}k")
    # #
    # elif "mecab" in args["tokenizer_type"]:
    #     input_corpus = INPUT_MECAB_TOKENIZED_CORPUS
    #     output_dir = os.path.join(OUTPUT_DIR, f"{tokenizer_type}_{composition_type}_sp-{int(args['vocab_size'])//1000}k")
    #
    # else:
    #     raise ValueError





    os.makedirs(output_dir, exist_ok=True)

    # save arguments info
    output_info_path = os.path.join(output_dir, "build_info.json")
    with open(output_info_path, "w", encoding="utf-8") as f:
        json.dump(args, f, indent=4)


    # train
    wp_tokenizer = train_wp(vocab_size=args["vocab_size"], files=input_file_paths, save_path=output_dir)    # tok.model



    # save
    # save_bert_vocab(output_dir=output_dir)  # vocab.txt
    # save_tok_vocab(output_dir=output_dir)   # tok.vocab
    save_wp_vocab(output_dir=output_dir)  # vocab.txt   # tok.vocab




    with open(os.path.join(args["tokenized_corpus_path"], "tok.json")) as f:
        tok_json = json.load(f)

    with open(os.path.join(output_dir, "tok.json"), "w") as f:
        json.dump(tok_json, f, indent=4)



    # # mecab config
    # tok_json = dict()
    # tok_json["dummy_letter"] = "⊸"
    # tok_json["space_symbol"] = "▃"
    # # tok_json["n_jobs"] =
    # # if "orig" in tokenizer_type:
    # #     tok_json["use_original"] = True
    # # elif "fixed" in tokenizer_type:
    # #     tok_json["use_original"] = False
    #
    # tok_json["tokenizer_type"] = tokenizer_type
    # tok_json["composition_type"] = composition_type
    #
    # if composition_type == "composed":
    #     tok_json["pure_decomposition"] = False
    #     tok_json["morphological"] = False
    #
    # elif composition_type == "decomposed_pure":
    #     tok_json["pure_decomposition"] = True
    #     tok_json["morphological"] = False
    #
    # elif composition_type == "decomposed_morphological":
    #     tok_json["pure_decomposition"] = False
    #     tok_json["morphological"] = True
    # else:
    #     tok_json["pure_decomposition"] = None
    #     tok_json["morphological"] = None
    #
    #
    # with open(os.path.join(output_dir, "tok.json"), "w") as f:
    #     json.dump(tok_json, f, indent=4)
