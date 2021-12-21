import os

from tokenizer import (
    # CharTokenizer,
    # JamoTokenizer,
    MeCabSentencePieceTokenizer_orig,
    MeCabSentencePieceTokenizer_fixed,
    MeCabSentencePieceTokenizer,
    MeCabWordPieceTokenizer,
    # MeCabTokenizer,
    MeCabTokenizer_orig,
    MeCabTokenizer_fixed,    # MeCabSentencePieceTokenizer_kortok,
    MeCabTokenizer_all,
    # MeCabTokenizer_kortok,
    SentencePieceTokenizer,
    WordPieceTokenizer,
    Vocab,
    # WordTokenizer,
)



def get_tokenizer(tokenizer_name: str, resource_dir: str, token_type, tokenizer_type: str , decomposition_type: str, space_symbol: str, dummy_letter: str, nfd: bool, grammatical_symbol: list = ["", ""]):
    tokenizer_dir = os.path.join(resource_dir, tokenizer_name)

    if tokenizer_name.startswith("sp-"):
        tokenizer = SentencePieceTokenizer(os.path.join(tokenizer_dir, "tok.model"))

    elif tokenizer_name.startswith("mecab_"):

        sp = SentencePieceTokenizer(os.path.join(tokenizer_dir, "tok.model"))

        if "orig" in tokenizer_name:
            mecab = MeCabTokenizer_orig(tokenizer_type=tokenizer_type, decomposition_type=decomposition_type, space_symbol=space_symbol, dummy_letter=dummy_letter)
            tokenizer = MeCabSentencePieceTokenizer_orig(mecab, sp, use_fixed=False) # mecab_sp_orig.py

        elif "fixed" in tokenizer_name:
            mecab = MeCabTokenizer_fixed(tokenizer_type=tokenizer_type, decomposition_type=decomposition_type, space_symbol=space_symbol, dummy_letter=dummy_letter)
            tokenizer = MeCabSentencePieceTokenizer_fixed(mecab, sp, use_fixed=True) # mecab_fixed.py


    elif tokenizer_name.startswith("eojeol") or tokenizer_name.startswith("morpheme"):
        wp = WordPieceTokenizer(os.path.join(tokenizer_dir, "bert_tokenizer.json"))
        mecab = MeCabTokenizer_all(token_type=token_type, tokenizer_type=tokenizer_type, decomposition_type=decomposition_type, space_symbol=space_symbol, dummy_letter=dummy_letter, nfd=nfd, grammatical_symbol=grammatical_symbol)
        tokenizer = MeCabWordPieceTokenizer(mecab=mecab, wp=wp) # mecab_wp.py


    else:
        raise ValueError("Wrong tokenizer name.")

    return tokenizer

