from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
from tokenizer.get_tokenizer import get_tokenizer

with open("/home/kist/rsync/namuwiki_20210301_with_preprocessing_v5_kss.txt") as f:
    corpus = f.readlines()

corpus = [line[:-1] for line in corpus]


tokenizer_name = "morpheme_mecab_orig_composed_grammatical_symbol_F_wp-64k"
resource_dir = "./resources/v6_without_dummy_letter_grammatical_symbol_F"
token_type = "morpheme"
tokenizer_type = "mecab_fixed"
decomposition_type = "composed"
space_symbol = ""
dummy_letter = ""
nfd = False
grammatical_symbol = ["", ""]
skip_sepcial_tokens = False


Tokenizer = get_tokenizer(tokenizer_name=tokenizer_name, resource_dir=resource_dir,
                          token_type=token_type,
                          tokenizer_type=tokenizer_type,
                          decomposition_type=decomposition_type,
                          space_symbol=space_symbol,
                          dummy_letter=dummy_letter, nfd=nfd,
                          grammatical_symbol=grammatical_symbol,
                          skip_special_tokens=skip_sepcial_tokens)

Tokenizer.tokenize("훌륭한 예시이다")


def tokenize_fun(text: str):
    tokenized = Tokenizer.tokenize(text)
    return tokenized

chr(9999)

tokenize_fun("훌륭한 예시이다")
tokenize_fun("훏✏뷁(ㄴㅇㄹ")

fn = partial(tokenize_fun)

threads = 8

with Pool(threads) as p:
    tokenized_corpus = p.map(fn, corpus[:9])    # 라인별로 mecab + BPE 토큰화한 코퍼스

51630038/2

tokenized_corpus = [Tokenizer.tokenize(line) for line in tqdm(corpus, position=0, leave=True)]

fn(corpus[0])