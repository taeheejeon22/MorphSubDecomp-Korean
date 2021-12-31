from typing import Dict, List, Tuple
import re

# p_kakao = re.compile(r"[^ㄱ-ㅎㅏ-ㅣ가-힣\x20-\x7F]*")  # 타 언어 문자, 특수 기호 제거
# 띄어쓰기
from quickspacer import Spacer
spacer = Spacer()

def load_data(file_path: str, label_to_index: Dict[str, int]) -> Tuple[List[str], List[str], List[int]]:
    """
    file_path에 존재하는 tsv를 읽어서 bert_data.InputIds 형태로 변경해주는 함수입니다.
    각각의 row를 bert input으로 바꾸어주기 위한 함수입니다.
    각 row는 아래처럼 구성되어야 합니다.
    1. sentence_a
    2. sentence_b
    3. label
    """
    sentence_as = []
    sentence_bs = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()[1:]):
            splitted = line.strip().split("\t")
            if len(splitted) != 5:
                #print(f"[ERROR] {repr(line)}, line {i}")
                continue
            # 띄어쓰기
            splitted[0] = spacer.space([splitted[0]])[0]
            splitted[1] = spacer.space([splitted[1]])[0]

            sentence_as.append(splitted[0])
            sentence_bs.append(splitted[1])
            labels.append(label_to_index[splitted[4]])
    # sentence_as = [re.sub(p_kakao, "", sentence) for sentence in sentence_as]
    # sentence_bs = [re.sub(p_kakao, "", sentence) for sentence in sentence_bs]
    return sentence_as, sentence_bs, labels
