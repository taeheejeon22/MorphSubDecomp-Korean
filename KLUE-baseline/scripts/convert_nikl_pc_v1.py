# v1
# 원문장별로 stratified하게 샘플링
# train set 줄임

# 모두의 말뭉치 유사 문장 판단 코퍼스를 tsv로 변환
# 1: human  0: machine

import json
import random
import pandas as pd

random.seed(42)

with open("./dataset/nlu_tasks/pc/NIKL_PC.json") as f:
    data = json.load(f)

# len(data)
# 
# type(data)
# 
# lst_data = list(data.items())
# len(lst_data)
# type(lst_data)
# lst_data[2]
# 
# data.keys()
# 
# data[]
# 
# 
real_data = data["data"]
# len(real_data)  # 17,959
# 
# for ix in range(len)
# 
# real_data[0].keys() # dict_keys(['sentence_id', 'paraphrases'])
# real_data[0]["sentence_id"]
# len(real_data[0]["paraphrases"])
# 
# 
# sentence_data["paraphrases"][0].keys()


# sentences = list()
# labels = list()

train_data = list()
dev_data = list()
test_data = list()


real_data[0]["paraphrases"]
random.shuffle(real_data[0]["paraphrases"])


# split
for ix in range(len(real_data)):
    sentence_data = real_data[ix]["paraphrases"]

    random.shuffle(sentence_data)   # 문장 순서 섞기

    train_data += sentence_data[0:len(sentence_data)-5]
    dev_data += sentence_data[len(sentence_data)-5:len(sentence_data)-3]
    test_data += sentence_data[len(sentence_data)-3:]



# generate labels
def generate_dataset(data):
    new_data = [(pair["form"], 1) if pair["generation"] == "human" else (pair["form"], 0) for pair in data]

    list_sent = [pair[0] for pair in new_data]
    list_label = [pair[1] for pair in new_data]

    return list_sent, list_label


train_sent, train_label = generate_dataset(data=train_data)
dev_sent, dev_label = generate_dataset(data=dev_data)
test_sent, test_label = generate_dataset(data=test_data)



#
train_df = pd.DataFrame({"sentence": train_sent, "label": train_label})
dev_df = pd.DataFrame({"sentence": dev_sent, "label": dev_label})
test_df = pd.DataFrame({"sentence": test_sent, "label": test_label})


train_df.to_csv("./dataset/nlu_tasks/pc/pc_train.tsv", sep="\t", header=None, index=None)
dev_df.to_csv("./dataset/nlu_tasks/pc/pc_dev.tsv", sep="\t", header=None, index=None)
test_df.to_csv("./dataset/nlu_tasks/pc/pc_test.tsv", sep="\t", header=None, index=None)






# 실수 체크
# [1 for sent in test_df["sentence"] if sent in train_df["sentence"]]
# [1 for sent in dev_df["sentence"] if sent in train_df["sentence"]]
# [1 for sent in dev_df["sentence"] if sent in test_df["sentence"]]