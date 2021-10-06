# data
# repository: https://dump.thewiki.kr/
# version: 20200302

# extractor
# https://github.com/jonghwanhyeon/namu-wiki-extractor

import gzip
import json
import pickle
import re
import tqdm

from koalanlp import API
from koalanlp.proc import SentenceSplitter
from koalanlp.Util import initialize

from namuwiki.extractor import extract_text


initialize(hnn='LATEST')
splitter = SentenceSplitter(API.HNN)


# load the raw corpus
file_path = "/home/kist/rsync/namuwiki_20200302.json"

with open(file_path, 'r', encoding='utf-8') as input_file:
    namu_wiki = json.load(input_file)


def preprocess(sent_lst):
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






# save as a string
all_texts = ""

for ix in tqdm.tqdm( range(len(namu_wiki)) ):
    document = namu_wiki[ix]
    plain_text = extract_text(document['text'])

    # split_text = plain_text.splitlines()        # 28
    # split_text = kss.split_sentences(plain_text)  # 36
    split_text = splitter(plain_text)   # 60

    preprocessed_text = preprocess(sent_lst=split_text) # 59

    concat_text = "\n".join(preprocessed_text)


    if plain_text != "":    # 빈 문서가 아닌 경우만 저장
        all_texts += (concat_text + "\n\n")



# save as a pickle
with gzip.open("./pretrain_corpus/namuwiki_20200302.pkl", "wb") as f:
    pickle.dump(all_texts, f)
