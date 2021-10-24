# kcbert 보캡 만들어 보기.
# 실패. 걍 버려.

import pandas as pd


with open("../20190101_20200611_v2.txt", "r") as f:
    data = f.readlines()

# with open("../20190101_20200611_v2.txt", "r") as f:
#     data0 = f.read()
# "\n\n" in data0 # False

data_pd = pd.Series(data)

seed = 22
sampled_data_pd = data_pd.sample(frac=0.1, replace=False, random_state=seed)

print(len(data_pd))    # 86246285
print(len(sampled_data_pd))    # 8624628


# 10% sample 저장
with open("../20190101_20200611_v2_sampled_" + str(seed) + ".txt", "w") as f:
    for ix in range(len(sampled_data_pd)):
        f.write(sampled_data_pd.iloc[ix])


from tokenizers import BertWordPieceTokenizer

tokenizer = BertWordPieceTokenizer(lowercase=False)
tokenizer.train(["../20190101_20200611_v2_sampled_" + str(seed) + ".txt"], vocab_size=32000, limit_alphabet=6000)
# tokenizer.train(["../20190101_20200611_v2.txt"], vocab_size=32000, limit_alphabet=3000) # 코퍼스 전체 학습

tokenizer.save("..")



with open("./resources/kcbert/origin/vocab_large.txt") as f:
    vc_kcbert_l = f.readlines()

with open("./resources/kcbert/origin/vocab_base.txt") as f:
    vc_kcbert_b = f.readlines()

idxs = [ix for ix in range(len(vc_kcbert_b)) if vc_kcbert_b[ix] in vc_kcbert_l]


with open("./resources/kcbert/reproduced/vocab_" + str(seed) + ".txt") as f:
    vc_our = f.readlines()


idxs = [ix for ix in range(len(vc_our)) if vc_our[ix] in vc_kcbert_b]


print(len(idxs))
print(len(idxs)/len(vc_our))