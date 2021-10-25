# kcbert 방식대로 해 보기
# 폐기돼서 안 씀.

from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer(lowercase=False)

tokenizer.train(files="/home/kist/Downloads/20190101_20200611_v2.txt", vocab_size=32000, limit_alphabet=3000)



import numpy as np

with open("/home/kist/Downloads/20190101_20200611_v2.txt", "r") as f:
    data = f.readlines()

idxs = np.random.choice(len(data), len(data)//10, replace=False)

sampled = data[idxs]

np_data = np.array(data)