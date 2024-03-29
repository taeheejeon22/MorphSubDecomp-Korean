from typing import Dict, List, Tuple
import re
from tasks.hsd.config import TrainConfig
from quickspacer import Spacer
# spacer = Spacer()

def load_data(file_path: str, label_to_index: Dict[str, int]) -> Tuple[List[str], List[int]]:
    """
    file_path에 존재하는 tsv를 읽어서 bert_data.InputIds 형태로 변경해주는 함수입니다.
    각각의 row를 bert input으로 바꾸어주기 위한 함수입니다.
    각 row는 아래처럼 구성되어야 합니다.
    1. sentence_a
    2. sentence_b
    3. label
    """
    sentence = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()[1:]):
            splitted = line.strip().split("\t")
            if len(splitted) != 4:
                continue
            sentence.append(splitted[0])
            labels.append(label_to_index[splitted[3]])

    return sentence, labels
