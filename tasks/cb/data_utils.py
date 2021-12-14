from typing import Dict, List, Tuple

import pandas as pd


def load_data(file_path: str, label_to_index: Dict[str, int]) -> Tuple[List[str], List[str], List[int]]:
    """
    file_path에 존재하는 tsv를 읽어서 bert_data.InputIds 형태로 변경해주는 함수입니다.
    각각의 row를 bert input으로 바꾸어주기 위한 함수입니다.
    각 row는 아래처럼 구성되어야 합니다.
    1. sentence_a
    2. sentence_b
    3. label
    """
    
    # sentence_as = []
    # sentence_bs = []
    # labels = []

    # with open(file_path, "r", encoding="utf-8") as f:
    #     for i, line in enumerate(f.readlines()[1:]):
    #         splitted = line.strip().split("\t")
    #         if len(splitted) != 3:
    #             #print(f"[ERROR] {repr(line)}, line {i}")
    #             continue
    #         sentence_as.append(splitted[0])
    #         sentence_bs.append(splitted[1])
    #         labels.append(label_to_index[splitted[2]])

    data = pd.read_csv(file_path)

    sentence_as = data["Discourse"].to_list()
    sentence_bs = data["Proposition"].to_list()
    labels = label_to_index(data["class_Restrict"]).to_list()

    return sentence_as, sentence_bs, labels


# label_to_index = {"Entailment": 0, "Contradict": 1}
# file_path ="/home/kist/Desktop/git_ee/ACL_tokenization/acl_tokenization/dataset/nlu_tasks/cb/NIKL_CB_2020_train.csv"
# label_to_index = {"neutral": 0, "entailment": 1, "contradiction": 2}  # KorNLI
# file_path ="/home/kist/Desktop/git_ee/ACL_tokenization/acl_tokenization/dataset/nlu_tasks/kornli/xnli.dev.ko.tsv"
