# wi

# data
# repository: https://dumps.wikimedia.org/kowiki/latest/
# version: 20210901

# extractor
# https://github.com/attardi/wikiextractor

import gzip
import os
import pickle
import re
from tqdm import tqdm

# from koalanlp import API
# from koalanlp.proc import SentenceSplitter
# from koalanlp.Util import initialize

import kss

# initialize(hnn='LATEST')
# splitter = SentenceSplitter(API.HNN)




# get the full path of files
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]






def preprocess(sent_lst):
    # for wiki ko
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
    # sent_lst = [re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9 ]+", "", sent) for sent in sent_lst]   # only for Hanguls, numbers, space  # without punctuation


    # # p_num_string = re.compile("[0-9,.]+")   # 숫자 string을 "N"으로 치환
    # p_num_string = re.compile("[0-9]+([,./]?[0-9])*")  # 숫자 string을 "N"으로 치환    # 1 2,000  3.5    2/3     2.2/4
    # sent_lst = [re.sub(p_num_string, "N", sent) for sent in sent_lst]   # 1,000년 > N년       2.3회 > N회

    # # re.sub(p_num_string, "N", "1,000년의 세월")
    # # re.sub(p_num_string, "N", "1.000년의 세월 200년의 삶")
    # # re.sub(p_num_string, "N", "1.000년의 세월 2년의 삶")


    # our
    p_multiple_spaces = re.compile("\s+")   # 무의미한 공백
    sent_lst = [re.sub(p_multiple_spaces, " ", sent) for sent in sent_lst]  # 무의미한 공백을 스페이스(" ")로 치환

    # p_only_N = re.compile("^N( N)*$")   # 숫자만 있는 문장 # 'N N N N'
    # sent_lst = [sent for sent in sent_lst if not p_only_N.search(sent)]   # 숫자만 있는 문장 제거


    # our
    sent_lst = [sent for sent in sent_lst if not re.search(r"^\s+$", sent)]    # 빈 문장 제거
    sent_lst = [sent.strip() for sent in sent_lst if sent != ""]    # 빈 문장 제거

    # sent_lst = [sent for sent in sent_lst if len(sent.split(" ")) > 1]  # 어절 길이가 1인 문장 제거. 학습할 이웃이 없을 것이라고 판단되므로. (형태소 분석하면 길이가 늘어날 수 있기는 함.)

    return sent_lst





# load the raw corpus by doc
all_docs = list()

above_paths = sorted(listdir_fullpath("../wiki_text"))   # ['../text/AD', '../text/AI', '../text/AG', '../text/AB', '../text/AC', '../text/AA', '../text/AH', '../text/AF', '../text/AE']

for ix in tqdm( range(len(above_paths)) ):
    sub_paths = sorted(os.listdir(above_paths[ix]))

    for jx in range(len(sub_paths)):
        file_path = os.path.join(above_paths[ix], sub_paths[jx])

        # with open(file_path, "r") as f:
        #     corpus_file = f.readlines()
        #     corpus_file = [re.sub(r"\n$", "", sentence) for sentence in corpus_file]    # 각 행 마지막 "\n" 제거

        with open(file_path, "r") as f:
            corpus_file = f.read()
            doc = corpus_file.split("</doc>")  # split by document

        all_docs += doc



# generate a corpus
all_texts = ""
all_texts_list = list()

p_punct = re.compile("[.!?]")

for ix in tqdm( range(len(all_docs)) ):
# for ix in tqdm(range( 100 )):
    split_text0 = all_docs[ix].splitlines() # "\n" 단위로 분리
    split_text1 = [kss.split_sentences(split) if p_punct.search(split) else [split] for split in split_text0 ]  # 문장 분리기로 분리
    split_text2 = [sent for sent_lst in split_text1 for sent in sent_lst] # flatten

    # all_texts_list += split_text2

    preprocessed_text = preprocess(split_text2)

    concat_text = "\n".join(preprocessed_text)

    if concat_text != "":    # 빈 문서가 아닌 경우만 저장
        all_texts += (concat_text + "\n\n")




# save as a pickle
with gzip.open("../wikiko_20210901_with_preprocessing.pkl", "wb") as f:
    pickle.dump(all_texts, f)

# save as a txt
with open("../wikiko_20210901_with_preprocessing_v2.txt", "w") as f:
    f.write(all_texts)


