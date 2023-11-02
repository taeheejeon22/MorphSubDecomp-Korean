# wiki-ko 덤프 파일 전처리
# 선행 조건: ./corpus/raw_corpus에 Wikiextractor를 이용해 추출된 위키 파일들이 위치해 있어야 함

import argparse
import os
import re
from tqdm import tqdm


# get the full path of files
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]



def preprocess(sent_lst):
    # https://github.com/ratsgo/embedding/blob/master/preprocess/dump.py 참조
    p_email =  re.compile("(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", re.UNICODE)
    p_url = re.compile("(ftp|http|https)?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", re.UNICODE)
    p_wiki_char = re.compile("(\\*$|:$|^파일:.+|^;)", re.UNICODE)
    p_wiki_space =  re.compile("(\\s|゙|゚|　)+", re.UNICODE)
    p_tag = re.compile("^<.+>?", re.UNICODE)

    sent_lst = [re.sub(p_email, " ", sentence) for sentence in sent_lst]
    sent_lst = [re.sub(p_url, " ", sentence) for sentence in sent_lst]
    sent_lst = [re.sub(p_wiki_char, " ", sentence) for sentence in sent_lst]
    sent_lst = [re.sub(p_wiki_space, " ", sentence) for sentence in sent_lst]
    sent_lst = [re.sub(p_tag, " ", sentence) for sentence in sent_lst]

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
    sent_lst = [sent for sent in sent_lst if not (sent.endswith(".") and len(sent.split(" ")) <= 3) ]   # 퇴임 이후.    어린 시절.  생애 후반.  등등의 짧은 라인 없애기

    return sent_lst



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")

    args = vars(parser.parse_args())
    print(args)

    # load the raw corpus by doc
    all_docs = list()

    above_paths = sorted(listdir_fullpath(args["input_path"]))   # ['../text/AD', '../text/AI', '../text/AG', '../text/AB', '../text/AC', '../text/AA', '../text/AH', '../text/AF', '../text/AE']

    for ix in tqdm( range(len(above_paths)) ):
        sub_paths = sorted(os.listdir(above_paths[ix]))

        for jx in range(len(sub_paths)):
            file_path = os.path.join(above_paths[ix], sub_paths[jx])

            with open(file_path, "r") as f:
                corpus_file = f.read()
                doc = corpus_file.split("</doc>")  # split by document

            all_docs += doc


    # generate a corpus
    all_texts = ""

    p_punct = re.compile("[.!?]")

    for ix in tqdm( range(len(all_docs)) ):
        split_text0 = all_docs[ix].splitlines() # "\n" 단위로 분리

        preprocessed_text = preprocess(split_text0)

        concat_text = "\n".join(preprocessed_text)

        if concat_text != "":    # 빈 문서가 아닌 경우만 저장
            # all_texts += (concat_text + "\n") # 문서 사이 안 띄우기. 즉 masked LM만 학습
            all_texts += (concat_text + "\n\n") # 문서 사이 띄우기


    # save as a txt
    with open(args["output_path"], "w") as f:    # v3: 짧은 행 제거. 문장 분리 x   + 문서 사이 띄우기
        f.write(all_texts)
