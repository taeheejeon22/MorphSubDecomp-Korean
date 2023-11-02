# namuwiki 덤프 파일 전처리
# 선행 조건: ./corpus/raw_corpus에 namuwiki 덤프 파일 (json 포맷) 존재해야 함

import argparse
import json
import re
import tqdm
from namuwiki.extractor import extract_text


def preprocess(sent_lst):
    # our
    p_paren_str = re.compile("\(.+?\)") # 괄호 문자열("(xxx)") 삭제용
    sent_lst = [re.sub(p_paren_str, "", sent) for sent in sent_lst] # 사람(인간)은 짐승(동물)이다 > 사람은 짐승이다

    # kortok
    # p_kakao = re.compile(r"[^가-힣\x20-\x7F]*") # 타 언어 문자, 특수 기호 제거
    p_kakao = re.compile(r"[^ㄱ-ㅎㅏ-ㅣ가-힣\x20-\x7F]*")  # 타 언어 문자, 특수 기호 제거    # 자모 낱글자 살리기
    sent_lst = [re.sub(p_kakao, "", sent) for sent in sent_lst]

    # our
    p_multiple_spaces = re.compile("\s+")   # 무의미한 공백
    sent_lst = [re.sub(p_multiple_spaces, " ", sent) for sent in sent_lst]  # 무의미한 공백을 스페이스(" ")로 치환

    # our
    sent_lst = [sent for sent in sent_lst if not re.search(r"^\s+$", sent)]    # 빈 문장 제거
    sent_lst = [sent.strip() for sent in sent_lst if sent != ""]    # 빈 문장 제거

    # our
    sent_lst = [sent for sent in sent_lst if len(sent.split(" ")) >= 3 ]   # 퇴임 이후.    어린 시절.  생애 후반.  등등의 짧은 라인 없애기

    return sent_lst




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")

    args = vars(parser.parse_args())
    print(args)

    file_path = args["input_path"]

    with open(file_path, 'r', encoding='utf-8') as input_file:
        namu_wiki = json.load(input_file)

    # save as a string
    all_texts = ""

    for ix in tqdm.tqdm( range(len(namu_wiki)) ):
        document = namu_wiki[ix]
        plain_text = extract_text(document['text'])
        split_text = plain_text.splitlines()        # 28

        preprocessed_text = preprocess(sent_lst=split_text)  # 59

        concat_text = "\n".join(preprocessed_text)

        if concat_text != "":    # 빈 문서가 아닌 경우만 저장
            # all_texts += (concat_text + "\n") # 문서 사이 안 띄우기. 즉 masked LM만 학습
            all_texts += (concat_text + "\n\n")  # 문서 사이 띄우기

    with open(args["output_path"], "w") as f:  # v3: 짧은 행 제거. 문장 분리 x   + 문서 사이 띄우기
        f.write(all_texts)
