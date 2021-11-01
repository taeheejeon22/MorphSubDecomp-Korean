# 모두의 말뭉치 유사 문장 판단 코퍼스를 tsv로 변환

import json
import pandas as pd


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


# id = list()
# sentence_1 = list()
sentences = list()
labels = list()


for ix in range(len(real_data)):
    sentence_data = real_data[ix]

    for jx in range(len(sentence_data["paraphrases"])):
        sentences.append(sentence_data["paraphrases"][jx]["form"])    # machine / human generated sentence

        label_orig = sentence_data["paraphrases"][jx]["generation"]
        if label_orig == "human":
            label = 1
        elif label_orig == "machine":
            label = 0
        else:
            print(label_orig)
            raise ValueError

        labels.append(label)  # label


df_data = pd.DataFrame({"sentence": sentences, "label": labels})


# split
train_df = df_data.sample(frac=0.8, random_state=42)
dev_test_df = df_data.drop(train_df.index)

dev_df = dev_test_df.sample(frac=0.5, random_state=42)
test_df = dev_test_df.drop(dev_df.index)

train_df.to_csv("./dataset/nlu_tasks/pc/pc_train.tsv", sep="\t", header=None, index=None)
dev_df.to_csv("./dataset/nlu_tasks/pc/pc_dev.tsv", sep="\t", header=None, index=None)
test_df.to_csv("./dataset/nlu_tasks/pc/pc_test.tsv", sep="\t", header=None, index=None)



# 실수 체크
# [1 for sent in test_df["sentence"] if sent in train_df["sentence"]]
# [1 for sent in dev_df["sentence"] if sent in train_df["sentence"]]
# [1 for sent in dev_df["sentence"] if sent in test_df["sentence"]]