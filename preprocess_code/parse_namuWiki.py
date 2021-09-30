# data
# repository: https://dump.thewiki.kr/
# version: 20200302

# extractor
# https://github.com/jonghwanhyeon/namu-wiki-extractor

import gzip
import json
import pickle
import tqdm

from namuwiki.extractor import extract_text


# load the raw corpus
file_path = "./pretrain_corpus/namuwiki_20200302.json"

with open(file_path, 'r', encoding='utf-8') as input_file:
    namu_wiki = json.load(input_file)


# save as a string
all_texts = ""

for ix in tqdm.tqdm( range(len(namu_wiki)) ):
    document = namu_wiki[ix]
    plain_text = extract_text(document['text'])
    all_texts += (plain_text + "\n")


# save as a pickle
with gzip.open("./pretrain_corpus/namuwiki_20200302.pkl", "wb") as f:
    pickle.dump(all_texts, f)
