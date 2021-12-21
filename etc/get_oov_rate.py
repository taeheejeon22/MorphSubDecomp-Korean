
with open("/home/kist/rsync/namuwiki_20200302_morpheme_mecab_fixed_composed.txt") as f:
    corpus = f.readlines()

corpus = [line[:-1] for line in corpus]


tokenizer_type = "mecab_fixed"
decompositon_type = "composed"
space_symbol = ""
dummy_letter = ""





if config.tokenizer.startswith("sp-"):
    tokenizer = SentencePieceTokenizer(os.path.join(tokenizer_dir, "tok.model"))
elif config.tokenizer.startswith("mecab_"):

    # mecab = MeCabTokenizer(os.path.join(tokenizer_dir, "tok.json"))
    # mecab = MeCabTokenizer_fixed(tokenizer_type=tokenizer_config["tokenizer_type"], decomposition_type=tokenizer_config["decomposition_type"], space_symbol=tokenizer_config["space_symbol"], dummy_letter=tokenizer_config["dummy_letter"])
    sp = SentencePieceTokenizer(os.path.join(tokenizer_dir, "tok.model"))

    if "orig" in config.tokenizer:
        mecab = MeCabTokenizer_orig(tokenizer_type=tokenizer_config["tokenizer_type"],
                                    decomposition_type=tokenizer_config["decomposition_type"],
                                    space_symbol=tokenizer_config["space_symbol"],
                                    dummy_letter=tokenizer_config["dummy_letter"])
        tokenizer = MeCabSentencePieceTokenizer_orig(mecab, sp, use_fixed=False)  # mecab_sp_orig.py

        # if config.token_type in ["eojeol", "morpheme"]: # token type 지정하는 resources v6~ 방식이면
        #     mecab = MeCabTokenizer_all(token_type=tokenizer_config["token_type"], tokenizer_type=tokenizer_config["tokenizer_type"], decomposition_type=tokenizer_config["decomposition_type"], space_symbol=tokenizer_config["space_symbol"], dummy_letter=tokenizer_config["dummy_letter"])
        #     tokenizer = MeCabSentencePieceTokenizer(mecab=mecab, sp=sp) # mecab_sp.py
        # elif config.token_type == "":   # 기존의 mecab_orig, mecab_fixed 사용
        #     mecab = MeCabTokenizer_orig(tokenizer_type=tokenizer_config["tokenizer_type"], decomposition_type=tokenizer_config["decomposition_type"], space_symbol=tokenizer_config["space_symbol"], dummy_letter=tokenizer_config["dummy_letter"])
        #     tokenizer = MeCabSentencePieceTokenizer_orig(mecab, sp, use_fixed=False) # mecab_sp_orig.py
    elif "fixed" in config.tokenizer:
        mecab = MeCabTokenizer_fixed(tokenizer_type=tokenizer_config["tokenizer_type"],
                                     decomposition_type=tokenizer_config["decomposition_type"],
                                     space_symbol=tokenizer_config["space_symbol"],
                                     dummy_letter=tokenizer_config["dummy_letter"])
        tokenizer = MeCabSentencePieceTokenizer_fixed(mecab, sp, use_fixed=True)  # mecab_fixed.py

        # if config.token_type in ["eojeol", "morpheme"]: # token type 지정하는 resources v6~ 방식이면
        #     mecab = MeCabTokenizer_all(token_type=tokenizer_config["token_type"], tokenizer_type=tokenizer_config["tokenizer_type"], decomposition_type=tokenizer_config["decomposition_type"], space_symbol=tokenizer_config["space_symbol"], dummy_letter=tokenizer_config["dummy_letter"])
        #     tokenizer = MeCabSentencePieceTokenizer(mecab=mecab, sp=sp) # mecab_sp.py
        #
        # elif config.token_type == "":  # 기존의 mecab_orig, mecab_fixed 사용
        #     mecab = MeCabTokenizer_fixed(tokenizer_type=tokenizer_config["tokenizer_type"], decomposition_type=tokenizer_config["decomposition_type"], space_symbol=tokenizer_config["space_symbol"], dummy_letter=tokenizer_config["dummy_letter"])
        #     tokenizer = MeCabSentencePieceTokenizer_fixed(mecab, sp, use_fixed=True) # mecab_fixed.py

elif config.tokenizer.startswith("eojeol") or config.tokenizer.startswith("morpheme"):
    # wp = WordPieceTokenizer(os.path.join(tokenizer_dir, "tok.model"))
    wp = WordPieceTokenizer(os.path.join(tokenizer_dir, "bert_tokenizer.json"))

    mecab = MeCabTokenizer_all(token_type=tokenizer_config["token_type"],
                               tokenizer_type=tokenizer_config["tokenizer_type"],
                               decomposition_type=tokenizer_config["decomposition_type"],
                               space_symbol=tokenizer_config["space_symbol"],
                               dummy_letter=tokenizer_config["dummy_letter"], nfd=tokenizer_config["nfd"],
                               grammatical_symbol=tokenizer_config["grammatical_symbol"])
    tokenizer = MeCabWordPieceTokenizer(mecab=mecab, wp=wp)  # mecab_wp.py