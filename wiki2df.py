# 아래 링크에서위키백과 다운로드 후
    # https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B2%A0%EC%9D%B4%EC%8A%A4_%EB%8B%A4%EC%9A%B4%EB%A1%9C%EB%93%9C
# wikiextractor (https://github.com/attardi/wikiextractor) 이용해서 추출한 다음에 이용할 것.

import os
import re

import pandas as pd
from tqdm import tqdm

# get the full path of files
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

# preprocessing for wiki
# https://github.com/ratsgo/embedding/blob/master/preprocess/dump.py 참고
def preprocess(sentence):
    p_email =  re.compile("(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)", re.UNICODE)
    p_url = re.compile("(ftp|http|https)?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", re.UNICODE)
    p_wiki_char = re.compile("(\\*$|:$|^파일:.+|^;)", re.UNICODE)
    p_wiki_space =  re.compile("(\\s|゙|゚|　)+", re.UNICODE)
    p_tag = re.compile("^<.+>?", re.UNICODE)

    p_paren_str = re.compile("\(.+?\)")  # 괄호 문자열("(xxx)") 삭제용
    # p_num_string = re.compile("[0-9]+")  # 숫자 string을 "N"으로 치환
    p_num_string = re.compile("[0-9]+([,./]?[0-9])+")  # 숫자 string을 "N"으로 치환    # 2,000  3.5    2/3     2.2/4

    p_only_N = re.compile("^N( N)*$")  # 숫자만 있는 문장 # 'N N N N'

    p_multiple_spaces = re.compile("\s+", re.UNICODE)

    preprocessed_sentence = re.sub(p_email, " ", sentence)
    preprocessed_sentence = re.sub(p_url, " ", preprocessed_sentence)
    preprocessed_sentence = re.sub(p_wiki_char, " ", preprocessed_sentence)
    preprocessed_sentence = re.sub(p_wiki_space, " ", preprocessed_sentence)
    preprocessed_sentence = re.sub(p_tag, " ", preprocessed_sentence)

    preprocessed_sentence = re.sub(p_paren_str, "", preprocessed_sentence)
    preprocessed_sentence = re.sub(p_num_string, "N", preprocessed_sentence)
    preprocessed_sentence = re.sub(p_only_N, "", preprocessed_sentence)

    preprocessed_sentence = re.sub(p_multiple_spaces, " ", preprocessed_sentence)

    return preprocessed_sentence


# re.sub(p_tag, "EE", "</ee><gg></fff>")


# main
all_corpus = list()

above_paths = sorted(listdir_fullpath("../text"))   # ['../text/AD', '../text/AI', '../text/AG', '../text/AB', '../text/AC', '../text/AA', '../text/AH', '../text/AF', '../text/AE']

for ix in tqdm( range(len(above_paths)) ):
    sub_paths = sorted(os.listdir(above_paths[ix]))

    for jx in range(len(sub_paths)):
        file_path = os.path.join(above_paths[ix], sub_paths[jx])

        with open(file_path, "r") as f:
            corpus_file = f.readlines()
            corpus_file = [re.sub(r"\n$", "", sentence) for sentence in corpus_file]    # 각 행 마지막 "\n" 제거

        corpus_file = [preprocess(sentence) for sentence in corpus_file]

        corpus_file = [sentence for sentence in corpus_file if not re.search(r"^\s*$", sentence)]  # 빈 행 제거

        all_corpus += corpus_file


# make a dataframe of sentences
df_wiki = pd.DataFrame({"Sentence": all_corpus})

# save
df_wiki.to_pickle("./output/wiki_ 01-Sep-2021 20:25.pkl", protocol=5)

# load
df_wiki = pd.read_pickle("./output/wiki_ 01-Sep-2021 20:25.pkl")

sum([len(sent.split(" ")) for sent in df_wiki["Sentence"]])

# 4,036,922 행. not 문장. 문서 자체가 들어간 경우도 있음.
# 67,947,691 어절