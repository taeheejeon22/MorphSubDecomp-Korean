from typing import Dict, List, Tuple
import re
# p_kakao = re.compile(r"[^ㄱ-ㅎㅏ-ㅣ가-힣\x20-\x7F]*")  # 타 언어 문자, 특수 기호 제거
# 띄어쓰기
from quickspacer import Spacer
spacer = Spacer()

def load_data(file_path: str, label_to_index: Dict[str, int]) -> Tuple[List[str], List[int]]:
    """
    file_path에 존재하는 tsv를 읽어서 bert_data.InputIds 형태로 변경해주는 함수입니다.
    각각의 row를 bert input으로 바꾸어주기 위한 함수입니다.
    각 row는 아래처럼 구성되어야 합니다.
    1. sentence
    3. label
    """
    sentences: List[str] = []
    labels: List[int] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()[1:]):
            splitted = line.strip().split("\t")
            if len(splitted) != 2:
                #print(f"[ERROR] {repr(line)}, line {i}")
                continue
            # 띄어쓰기
            splitted[0] = spacer.space([splitted[0]])[0]
            
            sentences.append(splitted[0])
            labels.append(label_to_index[splitted[1]])
    #sentences = [re.sub(p_kakao, "", s) for s in sentences]

    return sentences, labels
