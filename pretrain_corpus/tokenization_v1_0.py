import gzip
import os

import numpy as np
import pickle
import re
from tqdm import tqdm
# import jamo_functions_v1 as jamo    # mecab orig 사용 시 '잔다' VV+EC 등을 자모 분해하지 않음 '잔다'
# import jamo_functions_v2 as jamo    # mecab orig 사용 시 '잔다' VV+EC 등을 자모 분해함  'ㅈㅏㄴㄷㅏ'
# import jamo_functions_v3 as jamo
                                    # corpus v3
                                    # 자음만 있는 조사, 어미 종성에 위치하도록
                                    # 나 --ㄴ 고양이 이 ㄷㅏ-
                                    # '준 사람' > '주 --ㄴ 사람'

from konlpy.tag import Mecab


# sent = "필드 전투를 피해 채집 포인트에 도착한 후 열심히 아이템을 캐는 중에"
# jamo.str2jamo_morphological(sent, morpheme_analysis=True, use_original=use_original)


# preprocess
def preprocess(sent_lst):
    p_kakao = re.compile(r"[^가-힣\x20-\x7F]*")

    sent_lst = [re.sub(p_kakao, "", sent) for sent in sent_lst]

    # p_paren_str = re.compile("\(.+?\)") # 괄호 문자열("(xxx)") 삭제용
    # sent_lst = [re.sub(p_paren_str, "", sent) for sent in sent_lst] # 사람(인간)은 짐승(동물)이다 > 사람은 짐승이다

    # sent_lst = [re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9 ]+", "", sent) for sent in sent_lst]   # only for Hanguls, numbers, space  # without punctuation

    # # p_num_string = re.compile("[0-9,.]+")   # 숫자 string을 "N"으로 치환
    # p_num_string = re.compile("[0-9]+([,./]?[0-9])*")  # 숫자 string을 "N"으로 치환    # 1 2,000  3.5    2/3     2.2/4
    # sent_lst = [re.sub(p_num_string, "N", sent) for sent in sent_lst]   # 1,000년 > N년       2.3회 > N회

    # # re.sub(p_num_string, "N", "1,000년의 세월")
    # # re.sub(p_num_string, "N", "1.000년의 세월 200년의 삶")
    # # re.sub(p_num_string, "N", "1.000년의 세월 2년의 삶")

    # p_multiple_spaces = re.compile("\s+")   # 무의미한 공백
    # sent_lst = [re.sub(p_multiple_spaces, " ", sent) for sent in sent_lst]  # 무의미한 공백을 스페이스(" ")로 치환

    # p_only_N = re.compile("^N( N)*$")   # 숫자만 있는 문장 # 'N N N N'
    # sent_lst = [sent for sent in sent_lst if not p_only_N.search(sent)]   # 숫자만 있는 문장 제거

    # sent_lst = [sent for sent in sent_lst if not re.search(r"^\s+$", sent)]    # 빈 문장 제거
    # sent_lst = [sent.strip() for sent in sent_lst if sent != ""]    # 빈 문장 제거

    # sent_lst = [sent for sent in sent_lst if len(sent.split(" ")) > 1]  # 어절 길이가 1인 문장 제거. 학습할 이웃이 없을 것이라고 판단되므로. (형태소 분석하면 길이가 늘어날 수 있기는 함.)

    return sent_lst


# load corpus
def load_corpus():
    with gzip.open(corpus_path, "rb") as f:
        corpus = pickle.load(f)
        sent_lst = corpus.split("\n")   # 문장 단위로 분절

    del f, corpus

    # preprocess
    # p_paren_str = re.compile("\(.+?\)") # 괄호 문자열("(xxx)") 삭제용
    # sent_lst = [re.sub(p_paren_str, "", sent) for sent in sent_lst] # 사람(인간)은 짐승(동물)이다 > 사람은 짐승이다
    #
    # sent_lst = [re.sub(r"[^ㄱ-ㅎㅏ-ㅣ가-힣0-9 ]+", "", sent) for sent in sent_lst]   # only for Hanguls, numbers, space  # without punctuation
    #
    # # p_num_string = re.compile("[0-9,.]+")   # 숫자 string을 "N"으로 치환
    # p_num_string = re.compile("[0-9]+([,./]?[0-9])*")  # 숫자 string을 "N"으로 치환    # 1 2,000  3.5    2/3     2.2/4
    # sent_lst = [re.sub(p_num_string, "N", sent) for sent in sent_lst]   # 1,000년 > N년       2.3회 > N회
    #
    # # re.sub(p_num_string, "N", "1,000년의 세월")
    # # re.sub(p_num_string, "N", "1.000년의 세월 200년의 삶")
    # # re.sub(p_num_string, "N", "1.000년의 세월 2년의 삶")
    #
    # p_multiple_spaces = re.compile("\s+")   # 무의미한 공백
    # sent_lst = [re.sub(p_multiple_spaces, " ", sent) for sent in sent_lst]  # 무의미한 공백을 스페이스(" ")로 치환
    #
    # p_only_N = re.compile("^N( N)*$")   # 숫자만 있는 문장 # 'N N N N'
    # sent_lst = [sent for sent in sent_lst if not p_only_N.search(sent)]   # 숫자만 있는 문장 제거
    #
    # sent_lst = [sent for sent in sent_lst if not re.search(r"^\s+$", sent)]    # 빈 문장 제거
    # sent_lst = [sent.strip() for sent in sent_lst if sent != ""]    # 빈 문장 제거
    #
    # sent_lst = [sent for sent in sent_lst if len(sent.split(" ")) > 1]  # 어절 길이가 1인 문장 제거. 학습할 이웃이 없을 것이라고 판단되므로. (형태소 분석하면 길이가 늘어날 수 있기는 함.)

    sent_lst = preprocess(sent_lst=sent_lst)
    print(len(sent_lst)) # 46,540,361 46,350,425 46,241,968 45,242,757  45,070,098  44,846,837  44,715,408  38,698,423  38,942,110

    return sent_lst



def tokenization(sent_lst, token_type, composition_type, use_original):


    p_multiple_spaces = re.compile("\s+")  # 무의미한 공백

    # if morph_analysis == False: # 형태소 분석하지 않고 어절 그대로 쓴다면
    if token_type == "eojeol": # 형태소 분석하지 않고 어절 그대로 쓴다면
        if composition_type == "composed":
            # tokenized_corpus = [sent.split(" ") for sent in tqdm(sent_lst, position=0, leave=True)]
            tokenized_corpus = [re.sub(p_multiple_spaces, " ", sent).split(" ") for sent in tqdm(sent_lst, position=0, leave=True)]

        elif composition_type == "decomposed_pure":
            tokenized_corpus = [jamo.str2jamo(sent).split(" ") for sent in tqdm(sent_lst, position=0, leave=True) ]

        elif composition_type == "decomposed_morpheme": # 하셨다 > 하시었다
            def morpheme_normalization(sentence):
                mc = Mecab(use_original=use_original)
                return ["".join([mor_pos[0] for mor_pos in word]) for word in mc.pos(sentence, flatten=False)]

            tokenized_corpus = [morpheme_normalization(sentence=sent) for sent in tqdm(sent_lst, position=0, leave=True)]

        elif "decomposed_morphological" in composition_type:    # 하셨다 > 하ㅅㅣㅇㅓㅆㄷㅏ
            tokenized_corpus = [jamo.str2jamo_morphological(sent, morpheme_analysis=False, use_original=use_original).split(" ") for sent in tqdm(sent_lst, position=0, leave=True)]


            # tokenized_corpus = [jamo.str2jamo_morphological(sent, morpheme_analysis=False, use_original=use_original).split(" ") for sent in tqdm(sent_lst, position=0, leave=True)]
            #
            # tc = list()
            # for jx in tqdm(range(len(sent_lst)), position=0, leave=True):
            #     tc.append( jamo.str2jamo_morphological(sent_lst[jx], morpheme_analysis=False, use_original=use_original).split(" ") )
            #
            # jx = 3470180
            # jx = 3470774  # 영치기 영차
            # jamo.str2jamo_morphological(sent_lst[jx], morpheme_analysis=False, use_original=use_original).split(" ")


    # elif token_type == "morpheme":
    elif "morpheme" in token_type:
        if composition_type == "composed":
            mc = Mecab(use_original=use_original)
            # tokenized_corpus = [mc.morphs(sent_lst[ix]) for ix in tqdm( range(len(sent_lst)), position=0, leave=True )]
            tokenized_corpus = [mc.morphs(sent) for sent in tqdm(sent_lst, position=0, leave=True)]

        elif composition_type == "decomposed_pure":
            mc = Mecab(use_original=use_original)
            # tokenized_corpus = [jamo.str2jamo( " ".join(mc.morphs(sent_lst[ix]) ) ).split(" ") for ix in tqdm(range(len(sent_lst)), position=0, leave=True)]
            tokenized_corpus = [jamo.str2jamo( " ".join(mc.morphs(sent) ) ).split(" ") for sent in tqdm(sent_lst, position=0, leave=True)]

        elif composition_type == "decomposed_morphological":
            # tokenized_corpus = [jamo.str2jamo_morphological(sent_lst[ix], morpheme_analysis=True, use_original=use_original).split(" ") for ix in tqdm(range(len(sent_lst)), position=0, leave=True)]
            tokenized_corpus = [jamo.str2jamo_morphological(sent, morpheme_analysis=True, use_original=use_original).split(" ") for sent in tqdm(sent_lst, position=0, leave=True)]

            # [jamo.str2jamo_morphological(sent, morpheme_analysis=True, use_original=True).split(" ") for sent in tqdm(sent_lst[56:57], position=0, leave=True)][0][-24:-15]
            # [jamo.str2jamo_morphological(sent, morpheme_analysis=True, use_original=False).split(" ") for sent in tqdm(sent_lst[56:57], position=0, leave=True)][0][-24:-15]
            # # ['캐', 'ㅇㅓ-'],
            # ['ㄴㅡㄴ'],
            # [''],
            # ['중'],
            # ['에']]

            # jamo v1
            # 'ㅇㅡㄹ',
            # '카',
            # 'ㄴㅡㄴ',
            # '중',
            # 'ㅇㅔ-']]

    return tokenized_corpus


# tokenized_composed[:2]
# tokenized_decomposed_pure[:2]
# tokenized_decomposed_morphological[:2]


# tokenized_corpus = [mc.morphs(sent_lst[ix]) for ix in tqdm( range(len(sent_lst)), position=0, leave=True )]
#
# tcs = list()
# for sent in tqdm(sent_lst, position=0, leave=True ):
#     tcs.append( mc.morphs(sent))




# make directory
def make_directory(token_type, composition_type):
    save_dir = corpus_path + "namuwiki_" + token_type + "/" + composition_type
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True) # a Python version of $mkdir -p


# save as a txt
def save_decomposed_corpus(file_name, token_type, composition_type, corpus):
    save_path = corpus_path + "namuwiki_" + "/".join([token_type, composition_type])
    file_path = save_path + "/" + file_name

    with open(file_path, "w") as f:
        for ix in range(len(corpus)):
            # f.write("".join(corpus[ix]) + "\n")
            f.write(" ".join(corpus[ix]) + "\n")

    print(f"saved in {file_path}")



def main(sent_lst, token_type, composition_type, use_original):
    # 메모리 문제로 인해 분할 처리
    # tokenized_corpus = tokenization(sent_lst=sent_lst, composition_type=composition_type, morph_analysis=False, use_original=use_original)

    # iter = 5  # all
    iter = 4    # no_1_sent

    # for ix in range(iter):
    for ix in range(0,1):
        print(f"\niteration: {ix}\n")
        begin_idx = 10000000*ix
        end_idx = 10000000 * (ix + 1)

        # if token_type == "eojeol":
        #     morph_analysis = False
        # elif "morpheme" in token_type:
        #     morph_analysis = True

        if ix < iter-1:
            tokenized_corpus = tokenization(sent_lst=sent_lst[begin_idx:end_idx], token_type=token_type, composition_type=composition_type, use_original=use_original)

        elif ix == iter-1:
            tokenized_corpus = tokenization(sent_lst=sent_lst[begin_idx:], token_type=token_type, composition_type=composition_type, use_original=use_original)

        # make a directory to save the result
        make_directory(token_type=token_type, composition_type=composition_type)

        # save
        if use_original == True:
            mecab_type = "mecab_orig"
        elif use_original == False:
            mecab_type = "mecab_fixed"

        file_name = "namuwiki_20210301_tokenized_" + "_".join([composition_type, token_type, mecab_type, str(ix)]) + ".txt"
        save_decomposed_corpus(file_name=file_name, token_type=token_type, composition_type=composition_type, corpus=tokenized_corpus)




if __name__ == "__main__":
    # corpus_path = "../"
    corpus_path = "/home/user/rsync/namuwiki_20200302.pkl"

    sent_lst = load_corpus()
    
    with gzip.open("./namuwiki_20200302_preprocessed.pkl", "wb") as f:
        pickle.dump(sent_lst, f)
    # for saving time...
    # with gzip.open(corpus_path + "namuwiki_20210301_preprocessed_no_1_sent_enter.pkl", 'rb') as f:
    #     sent_lst = pickle.load(f)



    # ## 1. 어절 단위 토큰화
    # token_type = "eojeol"   # eojeol    morpheme_orig   morpheme_fixed

    # use_original = True
    # # mc = Mecab(use_original=use_original)

    #     # 1) 어절 # ['훌륭한', '사망', '플래그의', '예시이다']
    # composition_type = "composed"
    # main(sent_lst=sent_lst, token_type=token_type, composition_type=composition_type, use_original=use_original)

    #     # 2) 어절 + 자모 분해 pure    # ['ㅎㅜㄹㄹㅠㅇㅎㅏㄴ', 'ㅅㅏ-ㅁㅏㅇ', 'ㅍㅡㄹㄹㅐ-ㄱㅡ-ㅇㅢ-', 'ㅇㅖ-ㅅㅣ-ㅇㅣ-ㄷㅏ-']
    # composition_type = "decomposed_pure"
    # main(sent_lst=sent_lst, token_type=token_type, composition_type=composition_type, use_original=use_original)

    # #     # 3-1) 어절 + 형태소 분해 with mecab original    # mecab original은 어차피 원래 어절 복원 가능이니까, 형태소 다시 합쳐도 단어랑 같음;;;
    # # composition_type = "decomposed_morpheme"
    # # main(sent_lst=sent_lst, token_type=token_type, composition_type=composition_type, use_original=use_original)

    # #     # 3-2) 어절 + 형태소 분해 with mecab fixed   # ['훌륭하ㄴ', '사망', '플래그의', '예시이다']  # 성능 구려서 안 씀
    # # use_original = False
    # # composition_type = "decomposed_morpheme"
    # # main(sent_lst=sent_lst, token_type=token_type, composition_type=composition_type, use_original=use_original)

    #     # 4-1) 어절 + 자모 분해 morphological with mecab original # ['훌륭ㅎㅏㄴ', '사망', '플래그ㅇㅢ-', '예시이ㄷㅏ-']
    # use_original = True
    # composition_type = "decomposed_morphological_mecab_orig"
    # main(sent_lst=sent_lst, token_type=token_type, composition_type=composition_type, use_original=use_original)    #### iter 2 3470181/10000000에서 문제 발생

    #     # 4-2) 어절 + 자모 분해 morphological with mecab fixed    # ['훌륭하ㄴ--', '사망', '플래그ㅇㅢ-', '예시이ㄷㅏ-']
    # use_original = False
    # composition_type = "decomposed_morphological_mecab_fixed"
    # main(sent_lst=sent_lst, token_type=token_type, composition_type=composition_type, use_original=use_original)



    # ## 2. 형태소 단위 토큰화 1: mecab original
    # token_type = "morpheme_orig"

    # use_original = True
    # # mc = Mecab(use_original=use_original)

    #     # 1) 형태소    # ['훌륭', '한', '사망', '플래그', '의', '예시', '이', '다']
    # composition_type = "composed"
    # main(sent_lst=sent_lst, token_type=token_type, composition_type=composition_type, use_original=use_original)

    #     # 2) 형태소 + 자모 분해 pure  # ['ㅎㅜㄹㄹㅠㅇ', 'ㅎㅏㄴ', 'ㅅㅏ-ㅁㅏㅇ', 'ㅍㅡㄹㄹㅐ-ㄱㅡ-', 'ㅇㅢ-', 'ㅇㅖ-ㅅㅣ-', 'ㅇㅣ-', 'ㄷㅏ-']
    # composition_type = "decomposed_pure"
    # main(sent_lst=sent_lst, token_type=token_type, composition_type=composition_type, use_original=use_original)

    #     # 3) 형태소 + 자모 분해 morphological # ['훌륭', 'ㅎㅏㄴ', '사망', '플래그', 'ㅇㅢ-', '예시', '이', 'ㄷㅏ-']
    # composition_type = "decomposed_morphological"
    # main(sent_lst=sent_lst, token_type=token_type, composition_type=composition_type, use_original=use_original)



    # ## 3. 형태소 단위 토큰화 2: mecab fixed
    # token_type = "morpheme_fixed"

    # use_original = False
    # # mc = Mecab(use_original=use_original)

    #     # 1) 형태소    # ['훌륭', '하', 'ㄴ', '사망', '플래그', '의', '예시', '이', '다']
    # composition_type = "composed"
    # main(sent_lst=sent_lst, token_type=token_type, composition_type=composition_type, use_original=use_original)

    #     # 2) 형태소 + 자모 분해 pure   # ['ㅎㅜㄹㄹㅠㅇ', 'ㅎㅏ-', 'ㄴ--', 'ㅅㅏ-ㅁㅏㅇ', 'ㅍㅡㄹㄹㅐ-ㄱㅡ-', 'ㅇㅢ-', 'ㅇㅖ-ㅅㅣ-', 'ㅇㅣ-', 'ㄷㅏ-']
    # composition_type = "decomposed_pure"
    # main(sent_lst=sent_lst, token_type=token_type, composition_type=composition_type, use_original=use_original)

    #     # 3) 형태소 + 자모 분해 morphological  # ['훌륭', '하', 'ㄴ--', '사망', '플래그', 'ㅇㅢ-', '예시', '이', 'ㄷㅏ-']
    # composition_type = "decomposed_morphological"
    # main(sent_lst=sent_lst, token_type=token_type, composition_type=composition_type, use_original=use_original)
