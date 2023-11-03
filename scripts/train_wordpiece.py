# 토큰화된 코퍼스 파일 이용해 wordpiece 모델 생성
# 참고: https://huggingface.co/docs/tokenizers/python/latest/pipeline.html#all-together-a-bert-tokenizer-from-scratch

import argparse
import json
import os

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers import normalizers
from tokenizers.normalizers import StripAccents, NFD, NFC
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer


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
    # bert_tokenizer.save(os.path.join(save_path, "tok.model"))
    bert_tokenizer.save(os.path.join(save_path, "bert_tokenizer.json")) # klue 양식으로 맞추기

    return bert_tokenizer



def listdir_fullpath(d):
    return sorted([os.path.join(d, f) for f in os.listdir(d)])



# vocab.txt tok.vocab 만들기
def save_wp_vocab(output_dir: str):
    with open(os.path.join(output_dir, "bert_tokenizer.json"), "r", encoding="utf-8") as f:
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
    OUTPUT_DIR = "./resources"

    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--tokenized_corpus_path", type=str, default="")  # 토큰화한 코퍼스 경로

    args = vars(parser.parse_args())
    print(args)

    # tokenized file info.
    with open(os.path.join(args["tokenized_corpus_path"], "tok.json")) as f:
        tok_json = json.load(f)

    try:
        if tok_json["lexical_grammatical"] == False:
            token_type = tok_json["token_type"]
        elif tok_json["lexical_grammatical"] == True:
            token_type = "LG"
    except KeyError:
        token_type = tok_json["token_type"]

    tokenizer_type = tok_json["tokenizer_type"]
    decomposition_type = tok_json["decomposition_type"]

    grammatical_symbol = "F" if tok_json["grammatical_symbol"] == ["",""] else "T"


    print(token_type, "###############")
    print(tokenizer_type, '#################')


    # set output dir
    input_file_paths = [path for path in listdir_fullpath(args["tokenized_corpus_path"]) if path.endswith(".txt")]

    print(f"\n\nINPUT_CORPORA: {input_file_paths}\n\n")

    output_dir = os.path.join(OUTPUT_DIR, f"{token_type}_{tokenizer_type}_{decomposition_type}_grammatical_symbol_{grammatical_symbol}_wp-{args['vocab_size']}")

    os.makedirs(output_dir, exist_ok=True)

    # save arguments info
    output_info_path = os.path.join(output_dir, "build_info.json")
    with open(output_info_path, "w", encoding="utf-8") as f:
        json.dump(args, f, indent=4)

    # train
    wp_tokenizer = train_wp(vocab_size=args["vocab_size"], files=input_file_paths, save_path=output_dir)    # bert_tokenizer.json (구 tok.model)

    # save
    save_wp_vocab(output_dir=output_dir)  # vocab.txt   # tok.vocab

    with open(os.path.join(args["tokenized_corpus_path"], "tok.json")) as f:
        tok_json = json.load(f)

    with open(os.path.join(output_dir, "tok.json"), "w") as f:
        json.dump(tok_json, f, indent=4)
