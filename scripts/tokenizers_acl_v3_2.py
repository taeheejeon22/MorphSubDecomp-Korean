# v3_2:
# LG: 접사는 LEXICAL로 통합하지 않고 분리. 어휘 목록 지나치게 늘어나므로.
    # 이용되, 이용하, 이용받     # 이용 되  이용 하    이용 받
# 자소 분해는 기존 대로 조사, 어미에 대해서만 함. 따라서 접사가 분리되어도 LEXICAL과 동일한 자소 분해를 함.
    # lexical 합법 적 ㅇㅣ ㄷㅏ        ㅎㅏㅂㅂㅓㅂ ㅈㅓㄱ  이 다


# acv_v3_1:
# lexical_grammatical 기능 추가. 어절에서 문법 형태소만 분리
# 육식 동물에서는 -> 육식동물 에서 는
# 지정사/계사/서술격 조사도 문법 형태소에 포함


# acl_v3:
# transform_v3(): NFD로 분해하는 방식 추가

# acl_v2
# pure_decomposition, morphological -> decomposition type
# use_original -> tokenizer_type: str


# 원래 jamo_functions_v3.py 였음.
# v3
# corpus v3
    # 자음만 있는 조사, 어미 종성에 위치하도록
    # 나 --ㄴ 고양이 이 ㄷㅏ-
    # '준 사람' > '주 --ㄴ 사람'
# str2jamo_morphologicl(): flatten 기능 추가
# ['필드'], ['전투', 'ㄹㅡㄹ'], ['피하', 'ㅇㅏ-'], ['채집'], ['포인트', 'ㅇㅔ-'], ['도착', '하', '--ㄴ'], ['후'], ['열심히'], ['아이템', 'ㅇㅡㄹ'], ['캐', 'ㄴㅡㄴ'], ['중', 'ㅇㅔ-']]



# v2
# corpus v2
# mecab orig 사용 시 '잔다' VV+EC 등을 자모 분해함
# corpus v1
# mecab orig 사용 시 '잔다' VV+EC 등을 자모 분해 안 함


import re
import unicodedata

from itertools import chain
# from konlpy.tag import Mecab
# from tokenization._mecab import Mecab
from scripts._mecab import Mecab
from soynlp.hangle import compose, decompose, character_is_korean, character_is_complete_korean, character_is_moum, character_is_jaum

# from tokenization.jamo2str import moasseugi


doublespace_pattern = re.compile('\s+')


# class process_jamo():
class tokenizers():
    def __init__(self, dummy_letter: str = "", space_symbol: str = "", grammatical_symbol: list =["", ""], nfd: bool = True):
        self.dummy_letter = dummy_letter    # 초성/중성/종성 더미 문자
        self.space_symbol = space_symbol    # 띄어쓰기 더미 문자
        self.grammatical_symbol = grammatical_symbol    # 문법 형태소 표지 # ["⭧", "⭨"]
        self.grammatical_symbol_josa = grammatical_symbol[0]    # "⫸"   # chr(11000)
        self.grammatical_symbol_eomi = grammatical_symbol[1]    # "⭧"   # chr(11111)

        self.nfd = nfd # Unicode normalization Form D

        self.mc_orig = Mecab(use_original=True)
        self.mc_fixed = Mecab(use_original=False)

        # self.grammatical_pos = ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC", "EP", "EF", "EC", "ETN", "ETM"]
        # self.grammatical_pos_josa = ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC"]
        self.grammatical_pos_josa = ["JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JX", "JC", "VCP"]    # 계사/지정사/서술격 조사 '이' 포함
        self.grammatical_pos_eomi = ["EP", "EF", "EC", "ETN", "ETM"]
        self.grammatical_pos = self.grammatical_pos_josa + self.grammatical_pos_eomi

        self.punctuation_pos = ["SF", "SE", "SSO", "SSC", "SC", "SY"]

        # self.lexical_pos = ["NNG", "NNP", "NNB", "NNBC", "NR", "NP",
        #                     # "VV", "VA", "VX", "VCP", "VCN",
        #                     "VV", "VA", "VX", "VCN",    #     # 계사/지정사/서술격 조사 '이' 제외
        #                     "MM", "MAG", "MAJ", "IC",
        #                     "XPN", "XSN", "XSV", "XSA", "XR"]  # 접사도 포함시킴. 어미, 조사 아닌 것이므로.

        self.lexical_pos = ["NNG", "NNP", "NNB", "NNBC", "NR", "NP",
                            # "VV", "VA", "VX", "VCP", "VCN",
                            "VV", "VA", "VX", "VCN",    #     # 계사/지정사/서술격 조사 '이' 제외
                            "MM", "MAG", "MAJ", "IC",]
                            # "XPN", "XSN", "XSV", "XSA", "XR"]  # 접사도 포함시킴. 어미, 조사 아닌 것이므로.


        self.not_hangul_pos = ["SF", "SE", "SSO", "SSC", "SC", "SY", "SL", "SH", "SN"]



    ### general funcionts
    # 음절 분해용: 난 > ㄴㅏㄴ
    # https://github.com/ratsgo/embedding/blob/master/preprocess/unsupervised_nlputils.py
    def transform_v2(self, char):
        if char == ' ':  # 공백은 그대로 출력
            return char

        cjj = decompose(char)  # soynlp 이용해 분해

        # 자모 하나만 나오는 경우 처리 # ㄴ ㅠ
        try:
            if cjj.count(" ") == 2:
                if character_is_jaum(cjj[0]):  # 그 자모가 자음이면
                    cjj = (self.dummy_letter, self.dummy_letter, cjj[0])  # ('ㄴ', ' ', ' ') > ('-', 'ㄴ', '-')
                elif character_is_moum(cjj[0]):  # 그 자모가 모음이면
                    cjj = (self.dummy_letter, cjj[1], self.dummy_letter)  # (' ', 'ㅠ', ' ') > ('-', 'ㅠ', '-')
        except AttributeError:  # 혹시라도 한글 아닌 것이 들어올 경우 대비해
            pass

        if len(cjj) == 1:
            return cjj

        cjj_ = ''.join(c if c != ' ' else self.dummy_letter for c in cjj)
        return cjj_


    def transform_v3(self, char):
        if char == ' ':  # 공백은 그대로 출력
            return char

        cjj = unicodedata.normalize('NFD', char)    # unicode normalization (음절을 3개로)

        return cjj

        # # 자모 하나만 나오는 경우 처리 # ㄴ ㅠ
        # try:
        #     if cjj.count(" ") == 2:
        #         if character_is_jaum(cjj[0]):  # 그 자모가 자음이면
        #             cjj = (self.dummy_letter, self.dummy_letter, cjj[0])  # ('ㄴ', ' ', ' ') > ('-', 'ㄴ', '-')
        #         elif character_is_moum(cjj[0]):  # 그 자모가 모음이면
        #             cjj = (self.dummy_letter, cjj[1], self.dummy_letter)  # (' ', 'ㅠ', ' ') > ('-', 'ㅠ', '-')
        # except AttributeError:  # 혹시라도 한글 아닌 것이 들어올 경우 대비해
        #     pass
        #
        # unicodedata.normalize('NFD', char)




    # 문자열 일부 치환
    def str_substitution(self, orig_str, sub_idx, sub_str):
        lst_orig_str = [x for x in orig_str]
        # lst_sub_str = [x for x in sub_str]

        lst_orig_str[sub_idx] = sub_str

        # for ix in range(sub_span[-1] - sub_span[0]):
        #     lst_orig_str[sub_span[0]+ix] = sub_str[ix]

        return "".join(lst_orig_str)


    # for inserting space_symbol ("▃")
    # https://stackoverflow.com/questions/5920643/add-an-item-between-each-item-already-in-the-list
    def intersperse(self, lst, item):
        result = [[item]] * (len(lst) * 2 - 1)
        result[0::2] = lst
        return result


    # for inserting grammar_symbol ("⭧")
    def insert_grammar_symbol(self, mor_pos):
        # mor_pos: ('나', 'NP')
        # mor_pos: ('ㄴ', 'JX')
        # mor_pos: ('난', 'NP+JX')   # 무시

        pos = mor_pos[1]
        # if sum([1 for pos in pos.split("+") if pos in self.grammatical_pos]) >= 1: # 토큰 내에 문법 형태소 있으면


        if pos in self.grammatical_pos_josa:    # 조사이면
            new_mor = self.grammatical_symbol_josa + mor_pos[0]
        elif pos in self.grammatical_pos_eomi:  # 어미이면
            new_mor = self.grammatical_symbol_eomi + mor_pos[0]
        else:   # 어휘 형태소이면
            new_mor = mor_pos[0]

        return (new_mor, pos)



    # https://github.com/ratsgo/embedding/blob/master/preprocess/unsupervised_nlputils.py
    # def str2jamo(self, sent, jamo_morpheme=False):
    def str2jamo(self, sent, grammatical=False):
        # def transform(char):
        #     if char == ' ':
        #         return char
        #     cjj = decompose(char)
        #     if len(cjj) == 1:
        #         return cjj
        #     cjj_ = ''.join(c if c != ' ' else self.dummy_letter for c in cjj)
        #     return cjj_

        def transform_grammatical(char, grammatical):
            if char == ' ':
                return char
            cjj = decompose(char)

            if len(cjj) == 1:
                return cjj

            if grammatical == False:
                cjj_ = ''.join(c if c != ' ' else self.dummy_letter for c in cjj)
                return cjj_

            elif grammatical == True:
                cjj_without_blank = [x for x in cjj if x != " "] # remove " " from cjj

                if len(cjj_without_blank) == 1:   # if it is a jamo character (e.g. ㄴ, ㄹ, 'ㄴ'다)
                    cjj_ = self.dummy_letter * 2 + cjj_without_blank[0]

                elif len(cjj_without_blank) != 1:   # if it is a syllable character (e.g. 은, 을, 는다)
                    cjj_ = ''.join(c if c != ' ' else self.dummy_letter for c in cjj)

                return cjj_

            # cjj_ = ''.join(c if c != ' ' else self.dummy_letter for c in cjj)
            # return cjj_


        # if jamo_morpheme == False:
        #     sent_ = []
        #     for char in sent:
        #         if character_is_korean(char):
        #             sent_.append(transform(char))
        #         else:
        #             sent_.append(char)
        #     sent_ = doublespace_pattern.sub(' ', ''.join(sent_))
        #     return sent_
        #
        # if jamo_morpheme == True:   # for jamo morphemes like ㄴ, ㄹ, ...
        #     return self.dummy_letter*2 + sent   # '##ㄴ'


        sent_ = []
        for char in sent:
            if character_is_korean(char):
                sent_.append(transform_grammatical(char, grammatical=grammatical))
            else:
                sent_.append(char)
        sent_ = doublespace_pattern.sub(' ', ''.join(sent_))
        return sent_



    # https://github.com/ratsgo/embedding/blob/master/models/word_eval.py 참고
    def jamo2str(self, jamo):
        jamo_list, idx = [], 0
        while idx < len(jamo):
            if jamo[idx] == self.dummy_letter:  # -ㅠ- 처리용
                jamo_list.append(jamo[idx:idx + 3])
                idx += 3

            elif not character_is_korean(jamo[idx]):  #
                jamo_list.append(jamo[idx])
                idx += 1
            else:
                jamo_list.append(jamo[idx:idx + 3])
                idx += 3

        # jamo_list = list()
        # for ix in range(len(jamo)-2):
        #     if ix % 3 == 0: # 음절 시작이면
        #         jamo_list.append(jamo[ix:ix + 3])
        #     elif ix % 3 != 0: # 음절 시작이 아니면 다음 자모로 넘어가기
        #         continue
        #         # jamo_list.append(jamo[ix])
        #     else:
        #         raise Exception("check jamo2str()")

        word = ""
        for jamo_char in jamo_list:
            if len(jamo_char) == 1:
                word += jamo_char
            elif jamo_char[2] == self.dummy_letter:
                if jamo_char.count(self.dummy_letter) == 1:  # 일반적인 음절 문자 (ㅅㅏ-)
                    word += compose(jamo_char[0], jamo_char[1], " ")
                elif jamo_char.count(self.dummy_letter) == 2:  # 자모 하나만 있는 경우 (ㅋ--)
                    word += jamo_char.replace(self.dummy_letter, "")  # dummy letter 삭제하고 더하기
            elif (jamo_char[0] == self.dummy_letter) and (jamo_char[1] == self.dummy_letter):   # 문법 형태소 (--ㄹ, --ㄴ(다))
                previous_syllable = decompose(word[-1])
              
                word = word[:-1] + compose(previous_syllable[0], previous_syllable[1], jamo_char.replace(self.dummy_letter, "")) # dummy letter 삭제하고 앞 음절과 합치기
                # word += jamo_char.replace(self.dummy_letter, "")    # dummy letter 삭제하고 더하기


            else:
                word += compose(jamo_char[0], jamo_char[1], jamo_char[2])
        return word



    ######## tokenizer ###############
    ## 0. eojeol
    def eojeol_tokenizer(self, sent, decomposition_type: str):
                         # nfd: bool = False, morpheme_normalization: bool = False):
        # morpheme_normalization: 좋아해 -> 좋아하아


        p_multiple_spaces = re.compile("\s+")  # multiple blanks

        if decomposition_type == "composed":
        # if nfd == False:
        #     eojeol_tokenized = re.sub(p_multiple_spaces, " ", sent).split(" ")
            eojeol_tokenized = sent.split()

        elif decomposition_type == "decomposed_pure":
            if self.nfd == True:
                eojeol_tokenized = [self.transform_v3(eojeol) for eojeol in re.sub(p_multiple_spaces, " ", sent).split(" ")]
            elif self.nfd == False:
                eojeol_tokenized = [self.str2jamo(eojeol) for eojeol in re.sub(p_multiple_spaces, " ", sent).split(" ")]


        ## 폐기 ##
        # elif decomposition_type == "decomposed_lexical":
        #     mc = Mecab(use_original=False)
        #     if self.nfd == True:
        #         # 어휘 형태소만 unicode NFD 적용
        #         # eojeol_tokenized = ["".join([mor_pos[0] if (mor_pos[1] in self.grammatical_pos) else (self.grammatical_symbol + self.transform_v3(mor_pos[0]) ) for mor_pos in word]) for word in mc.pos(re.sub(p_multiple_spaces, " ", sent), flatten=False, join=False, coda_normalization=False) ]
        #         eojeol_tokenized = ["".join([self.transform_v3(mor_pos[0]) if (not mor_pos[1] in self.grammatical_pos) else (self.grammatical_symbol + mor_pos[0] ) for mor_pos in word]) for word in mc.pos(re.sub(p_multiple_spaces, " ", sent), flatten=False, join=False, coda_normalization=False) ]
        #
        #     elif self.nfd == False:
        #         # eojeol_tokenized = ["".join([mor_pos[0] if (mor_pos[1] in self.grammatical_pos) else (self.grammatical_symbol + self.str2jamo(mor_pos[0]) ) for mor_pos in word]) for word in mc.pos(re.sub(p_multiple_spaces, " ", sent), flatten=False, join=False, coda_normalization=True) ]
        #         eojeol_tokenized = ["".join([self.str2jamo(mor_pos[0]) if (not mor_pos[1] in self.grammatical_pos) else (self.grammatical_symbol + mor_pos[0] ) for mor_pos in word]) for word in mc.pos(re.sub(p_multiple_spaces, " ", sent), flatten=False, join=False, coda_normalization=True) ]
        #
        # elif decomposition_type == "decomposed_grammatical":
        #     mc = Mecab(use_original=False)
        #     if self.nfd == True:
        #         # 문법 형태소만 unicode NFD 적용
        #         eojeol_tokenized = ["".join([mor_pos[0] if (not mor_pos[1] in self.grammatical_pos) else (self.grammatical_symbol + self.transform_v3(mor_pos[0]) ) for mor_pos in word]) for word in mc.pos(re.sub(p_multiple_spaces, " ", sent), flatten=False, join=False, coda_normalization=False) ]
        #     elif self.nfd == False:
        #         eojeol_tokenized = ["".join([mor_pos[0] if (not mor_pos[1] in self.grammatical_pos) else (self.grammatical_symbol + self.str2jamo(mor_pos[0]) ) for mor_pos in word]) for word in mc.pos(re.sub(p_multiple_spaces, " ", sent), flatten=False, join=False, coda_normalization=True) ]


        return eojeol_tokenized


    ## 1. morpheme
    # def mecab_tokenizer(self, sent: str, use_original: bool, pure_decomposition: bool, morphological: bool = False):

    # def mecab_tokenizer(self, sent: str, tokenizer_type: str, decomposition_type: str):

    def mecab_tokenizer(self, sent: str, token_type: str, tokenizer_type: str, decomposition_type: str, flatten: bool = True, lexical_grammatical: bool = False):
    # def mecab_tokenizer(self, sent: str, token_type: str,  tokenizer_type: str = "mecab_fixed", decomposition_type: str = "composed", flatten: bool = True):
        assert (tokenizer_type in ["none", "mecab_orig", "mecab_fixed"] ), 'check the tokenizer type!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
        # assert (decomposition_type in ["composed", "decomposed_pure", "decomposed_morphological", "composed_nfd", "decomposed_pure_nfd", "decomposed_morphological_nfd"] ), 'check the decomposition type!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'

        # eojeol tokenization
        if token_type == "eojeol":
        # if tokenizer_type == "none":
            mecab_tokenized = self.eojeol_tokenizer(sent=sent, decomposition_type=decomposition_type)

        # morpheme tokenization
        elif token_type == "morpheme":
            if tokenizer_type == "mecab_orig":
                use_original = True
            elif tokenizer_type == "mecab_fixed":
                use_original = False

            if decomposition_type == "composed":
                mecab_tokenized = self.mecab_composed_decomposed_pure(sent=sent, lexical_grammatical=lexical_grammatical, use_original=use_original, pure_decomposition=False, nfd=self.nfd, flatten=flatten)
            # elif decomposition_type == "composed_nfd":  # coda normalization 안 하고, mecab 원래 출력대로 종성 문자 쓰기
            #     mecab_tokenized = self.mecab_composed_decomposed_pure(sent=sent, use_original=use_original, pure_decomposition=False, nfd=True)

            elif decomposition_type == "decomposed_pure":
                mecab_tokenized = self.mecab_composed_decomposed_pure(sent=sent, lexical_grammatical=lexical_grammatical, use_original=use_original, pure_decomposition=True, nfd=self.nfd, flatten=flatten)
                # mecab_tokenized = self.mecab_composed_decomposed_pure(sent=sent, use_original=use_original, pure_decomposition=True, nfd=False)
            # elif decomposition_type == "decomposed_pure_nfd":
            #     mecab_tokenized = self.mecab_composed_decomposed_pure(sent=sent, use_original=use_original, pure_decomposition=True, nfd=True)

            # elif decomposition_type == "decomposed_morphological":
            #     mecab_tokenized = self.mecab_with_morphological_decomposition(sent=sent, use_original=use_original, nfd=self.nfd)
            #     # mecab_tokenized = self.mecab_with_morphological_decomposition(sent=sent, use_original=use_original, nfd=False)
            # # elif decomposition_type == "decomposed_morphological_nfd":
            # #     mecab_tokenized = self.mecab_with_morphological_decomposition(sent=sent, use_original=use_original, nfd=True)

            elif decomposition_type == "decomposed_lexical":
                mecab_tokenized = self.mecab_with_morphological_decomposition(sent=sent, lexical_grammatical=lexical_grammatical, use_original=use_original, nfd=self.nfd, lexical_or_grammatical="lexical", flatten=flatten)

            elif decomposition_type == "decomposed_grammatical":
                mecab_tokenized = self.mecab_with_morphological_decomposition(sent=sent, lexical_grammatical=lexical_grammatical, use_original=use_original, nfd=self.nfd, lexical_or_grammatical="grammatical", flatten=flatten)

        return mecab_tokenized


    # 1-1. composed & decomposed_pure
    def mecab_composed_decomposed_pure(self, sent: str, use_original: bool, pure_decomposition: bool, nfd: bool, flatten: bool = True, lexical_grammatical: bool = False):
        # 문법 형태소만 분리: 육식동물 에서 는
        if lexical_grammatical == True:
            mor_poss = self.mecab_grammatical_tokenizer(sent=sent, nfd=nfd)

        # 순수 형태소 분석: 육식 동물 에서 는
        elif lexical_grammatical == False:
            if use_original == True:
                if nfd == False:
                    mor_poss = self.mc_orig.pos(sent, flatten=False, coda_normalization=True)  # [[('넌', 'NP+JX')], [('날', 'NNG')], [('좋', 'VA'), ('아', 'EC'), ('해', 'VV+EC')]]

                elif nfd == True:
                    mor_poss = self.mc_orig.pos(sent, flatten=False, coda_normalization=False)  # [[('넌', 'NP+JX')], [('날', 'NNG')], [('좋', 'VA'), ('아', 'EC'), ('해', 'VV+EC')]]

            else:
            # elif use_original == False:
                if nfd == False:
                    mor_poss = self.mc_fixed.pos(sent, flatten=False, coda_normalization=True)  # [[('너', 'NP'), ('ㄴ', 'JX')], [('날', 'NNG')], [('좋', 'VA'), ('아', 'EC'), ('하', 'VV'), ('아', 'EC')]]
                    # mor_poss = self.mc_orig.pos(sent, flatten=False, coda_normalization=True)  # [[('너', 'NP'), ('ㄴ', 'JX')], [('날', 'NNG')], [('좋', 'VA'), ('아', 'EC'), ('하', 'VV'), ('아', 'EC')]]
                elif nfd == True:
                    mor_poss = self.mc_fixed.pos(sent, flatten=False, coda_normalization=False)  # [[('너', 'NP'), ('ㄴ', 'JX')], [('날', 'NNG')], [('좋', 'VA'), ('아', 'EC'), ('하', 'VV'), ('아', 'EC')]]
                    # mor_poss = self.mc_orig.pos(sent, flatten=False, coda_normalization=False)  # [[('너', 'NP'), ('ㄴ', 'JX')], [('날', 'NNG')], [('좋', 'VA'), ('아', 'EC'), ('하', 'VV'), ('아', 'EC')]]


        # if pure_decomposition == False:
        #     pass
        # elif pure_decomposition == True:
        #     [ [ self.str2jamo(mor_pos, jamo_morpheme=False) if not mor_poss[-1] in self.grammatical_pos else self.str2jamo(mor_pos, jamo_morpheme=True) for mor_pos in word] for word in mor_poss]



        # insert grammatical symbol
        if len(self.grammatical_symbol) > 0:   # grammatical_symbol 사용하면
            mor_poss = [[self.insert_grammar_symbol(mor_pos=mor_pos) for mor_pos in word] for word in mor_poss]



        # remove pos tags
        if pure_decomposition == False:
            mors = [[mor_pos[0] for mor_pos in word] for word in mor_poss]  # [['너', 'ㄴ'], ['날'], ['좋', '아', '하', '아']]

        elif pure_decomposition == True:
            # mors = [ [ self.str2jamo(mor_pos[0], jamo_morpheme=False) if not mor_pos[-1] in self.grammatical_pos else self.str2jamo(mor_pos[0], jamo_morpheme=True) for mor_pos in word] for word in mor_poss]
            # mors = [ [ self.str2jamo(mor_pos[0], jamo_morpheme=True) if (mor_pos[-1] in self.grammatical_pos and len(mor_pos[0]) == 1 and character_is_jaum(mor_pos[0]) ) else self.str2jamo(mor_pos[0], jamo_morpheme=False) for mor_pos in word] for word in mor_poss]

            if nfd == False:
                mors = [ [ self.str2jamo(mor_pos[0], grammatical=True)  if (mor_pos[-1] in self.grammatical_pos ) else self.str2jamo(mor_pos[0], grammatical=False) for mor_pos in word] for word in mor_poss]
                                                                        # convert jamo morpheme like ㄴ, ㄹ into ##ㄴ, ##ㄹ
            elif nfd == True:
                mors = [[self.transform_v3(mor_pos[0]) for mor_pos in word] for word in mor_poss]


            # ee = list()
            # for word in mor_poss:
            #     for mor_pos in word:
            #         if (mor_pos[-1] in self.grammatical_pos and character_is_jaum(mor_pos[0]) ):
            #             ee.append(self.str2jamo(mor_pos[0], jamo_morpheme=True))
            #         else:
            #             ee.append(self.str2jamo(mor_pos[0], jamo_morpheme=False))


        if flatten == True:
            mecab_tokenized = list(chain.from_iterable(self.intersperse(mors, self.space_symbol)))  # ['너', 'ㄴ', '▃', '날', '▃', '좋', '아', '하', '아']

            if self.space_symbol == "": # 스페이스 심벌 안 쓴다면
                mecab_tokenized = [token for token in mecab_tokenized if token != ""]    # 빈 토큰 제외


        elif flatten == False:
            mecab_tokenized = self.intersperse(mors, self.space_symbol)  # [['너', 'ㄴ'], ['▃'], ['날'], ['▃'], ['좋', '아', '하', '아']]

            if self.space_symbol == "": # 스페이스 심벌 안 쓴다면
                mecab_tokenized = [token for token in mecab_tokenized if token != [""]]    # 빈 토큰 제외

        return mecab_tokenized

        # if pure_decomposition == False:
        #     return mecab_tokenized  # ['너', 'ㄴ', '▃', '날', '▃', '좋', '아', '하', '아']
        #
        # elif pure_decomposition == True:
        #     return [self.str2jamo(token) for token in mecab_tokenized]  # ['ㄴㅓ#', 'ㄴ##', '▃', 'ㄴㅏㄹ', '▃', 'ㅈㅗㅎ', 'ㅇㅏ#', 'ㅎㅏ#', 'ㅇㅏ#']

        ## decomposed_morphological용
        # 형태소 분석 과정이 있기 때문에 조금씩 노이즈 발생



    ## 1-2. decomposition morphological
    def mecab_with_morphological_decomposition(self, sent: str, use_original: bool, nfd: bool = False, lexical_or_grammatical: str = "lexical", flatten: bool = True, lexical_grammatical: bool = False):
        '''
        :param sent: 자모 변환할 문장      '너를 좋아해'
        :param morpheme_analysis:
            False: 자모 변환만 수행    (어절 토큰화 문장을 자모로 변환하는 데에 그대로 이용 가능)
            True: 형태소 분석 + 자모 변환
        :param use_original: konlpy original mecab 쓸지
        :param nfd: unicode NFD 적용해서 분해할지.
        :return: 자모 변환된 문장          '너ㅡㄹ 좋아해' or '너 ㄹㅡㄹ 좋아해'
        '''

        # 문법 형태소만 분리: 육식동물 에서 는
        if lexical_grammatical == True:
            mor_poss = self.mecab_grammatical_tokenizer(sent=sent, nfd=nfd)

        # 순수 형태소 분석: 육식 동물 에서 는
        elif lexical_grammatical == False:
            if use_original == True:
                if nfd == False:
                    mor_poss = self.mc_orig.pos(sent, flatten=False, coda_normalization=True)  # 형태소 분석
                elif nfd == True:
                    mor_poss = self.mc_orig.pos(sent, flatten=False, coda_normalization=False)  # 형태소 분석
            elif use_original == False:
                if nfd == False:
                    mor_poss = self.mc_fixed.pos(sent, flatten=False, coda_normalization=True)  # 형태소 분석
                elif nfd == True:
                    mor_poss = self.mc_fixed.pos(sent, flatten=False, coda_normalization=False)  # 형태소 분석


        new_sent = list()
        for ix in range(len(mor_poss)):
            eojeol = mor_poss[ix]  # [('나', 'NP'), ('는', 'JX')]

            new_eojeol = list()  # ['나', 'ㄴㅡㄴ']
            for jx in range(len(eojeol)):
                morpheme, pos = eojeol[jx]  # '너', 'NP'


                # if lexical_or_grammatical == "lexical":
                #     # 어휘 형태소가 아니면
                #     if sum([1 for pos in pos.split("+") if pos in self.lexical_pos]) < 1:  # 잔다 VV+EC 등을 분해함
                #         decomposed_morpheme = self.grammatical_symbol + morpheme[:]
                #
                #     # 어휘 형태소이면
                #     elif sum([1 for pos in pos.split("+") if pos in self.lexical_pos]) >= 1:  # 잔다 VV+EC 등을 분해함
                #         if nfd == False:
                #             decomposed_morpheme = "".join(
                #                 [self.transform_v2(char) if character_is_korean(char) else char for char in morpheme])  # 한 -> ㅎㅏㄴ
                #         elif nfd == True:
                #             decomposed_morpheme = "".join(
                #                 [self.transform_v3(char) if character_is_korean(char) else char for char in morpheme])  # 는 -> 는  # len("는"): 3

                if lexical_or_grammatical == "lexical":
                    # 문법 형태소가 들어 있으면
                    # if sum([1 for pos in pos.split("+") if pos in self.grammatical_pos]) >= 1:
                    #     decomposed_morpheme = self.grammatical_symbol + morpheme[:]

                    if pos in self.grammatical_pos_josa: # 조사이면
                        decomposed_morpheme = self.grammatical_symbol_josa + morpheme[:]
                    elif pos in self.grammatical_pos_eomi: # 어미이면
                        decomposed_morpheme = self.grammatical_symbol_eomi + morpheme[:]

                    else: # 어휘 형태소 혹은 혼종('난/NP+JX')이면
                        if nfd == False:
                            decomposed_morpheme = "".join(
                                [self.transform_v2(char) if character_is_korean(char) else char for char in morpheme])  # 한 -> ㅎㅏㄴ
                        elif nfd == True:
                            decomposed_morpheme = "".join(
                                [self.transform_v3(char) if character_is_korean(char) else char for char in morpheme])  # 는 -> 는  # len("는"): 3


                elif lexical_or_grammatical == "grammatical":
                    # 문법 형태소가 아니면
                    # if sum([1 for pos in pos.split("+") if pos in self.grammatical_pos]) < 1:  # 잔다 VV+EC 등을 분해함
                    #     decomposed_morpheme = morpheme[:]
                    if not (pos in self.grammatical_pos):
                        decomposed_morpheme = morpheme[:]

                    else:   # 문법 형태소이면

                        if pos in self.grammatical_pos_josa:   # 조사이면
                            grammatical_symbol = self.grammatical_symbol_josa[:]
                        elif pos in self.grammatical_pos_eomi:   # 어미이면
                            grammatical_symbol = self.grammatical_symbol_eomi[:]


                        if nfd == False:
                            decomposed_morpheme = grammatical_symbol + "".join(
                                [self.transform_v2(char) if character_is_korean(char) else char for char in morpheme])  # 한 -> ㅎㅏㄴ
                        elif nfd == True:
                            decomposed_morpheme = grammatical_symbol + "".join(
                                [self.transform_v3(char) if character_is_korean(char) else char for char in morpheme])  # 는 -> 는  # len("는"): 3

                    # 혼종(난/NP+JX)도 처리하는 방식
                    # elif sum([1 for pos in pos.split("+") if pos in self.grammatical_pos]) >= 1:  # 잔다 VV+EC 등을 분해함
                    #     if nfd == False:
                    #         decomposed_morpheme = self.grammatical_symbol + "".join(
                    #             [self.transform_v2(char) if character_is_korean(char) else char for char in morpheme])  # 한 -> ㅎㅏㄴ
                    #     elif nfd == True:
                    #         decomposed_morpheme = self.grammatical_symbol + "".join(
                    #             [self.transform_v3(char) if character_is_korean(char) else char for char in morpheme])  # 는 -> 는  # len("는"): 3


                new_eojeol.append(decomposed_morpheme)

            new_sent.append(new_eojeol)

            # if morpheme_tokenization == False:  # 형태소 토큰화 없이 어절 그대로 자모로 변환만 한다면
            #     # if flatten == True:
            #     #     new_sent.append("".join(new_eojeol))
            #     # elif flatten == False:
            #     new_sent.append(new_eojeol)
            # elif morpheme_tokenization == True:  # 형태소 토큰화 + 자모 변환 한다면
            #     # if flatten == True:
            #     #     new_sent += new_eojeol
            #     # elif flatten == False:
            #     new_sent.append(new_eojeol)

        # if flatten == True:
        #     new_sent = doublespace_pattern.sub(" ", " ".join(new_sent))
        # elif flatten == False:
        #     pass

        if flatten == True:
            new_sent_with_special_token = list(chain.from_iterable(self.intersperse(new_sent, self.space_symbol)))    # ['너', 'ㄴ', '▃', '날', '▃', '좋', '아', '하', '아']

            if self.space_symbol == "": # 스페이스 심벌 안 쓴다면
                new_sent_with_special_token = [token for token in new_sent_with_special_token if token != ""]    # 빈 토큰 제외

        elif flatten == False:
            new_sent_with_special_token = self.intersperse(new_sent, self.space_symbol)  # [['너', 'ㄴ'], ['▃'], ['날'], ['▃'], ['좋', '아', '하', '아']]

            if self.space_symbol == "":  # 스페이스 심벌 안 쓴다면
                new_sent_with_special_token = [token for token in new_sent_with_special_token if token != [""]]  # 빈 토큰 제외

        return new_sent_with_special_token


    # ## decomposed_morphological용
    #     # 형태소 분석 과정이 있기 때문에 조금씩 노이즈 발생
    # def str2jamo_morphological(self, sent, morpheme_tokenization, use_original, flatten=True):
    #     '''
    #     :param sent: 자모 변환할 문장      '너를 좋아해'
    #     :param morpheme_analysis:
    #         False: 자모 변환만 수행    (어절 토큰화 문장을 자모로 변환하는 데에 그대로 이용 가능)
    #         True: 형태소 분석 + 자모 변환
    #     :param use_original: konlpy original mecab 쓸지
    #     :return: 자모 변환된 문장          '너ㅡㄹ 좋아해' or '너 ㄹㅡㄹ 좋아해'
    #     '''
    #
    #     if use_original == True:
    #         mors_ejs_in_sent = self.mc_orig.pos(sent, flatten=False) # 형태소 분석
    #     elif use_original == False:
    #         mors_ejs_in_sent = self.mc_fixed.pos(sent, flatten=False)  # 형태소 분석
    #
    #     new_sent = list()
    #     for ix in range(len(mors_ejs_in_sent)):
    #         eojeol = mors_ejs_in_sent[ix]   # [('나', 'NP'), ('는', 'JX')]
    #
    #         new_eojeol = list()     # ['나', 'ㄴㅡㄴ']
    #         for jx in range(len(eojeol)):
    #             morpheme, pos = eojeol[jx]    # '너', 'NP'
    #
    #             # 문법 형태소가 아니면
    #             # if not pos in grammatical_pos:    # 잔다 VV+EC 등을 분해하지 않음
    #             if sum([1 for pos in pos.split("+") if pos in self.grammatical_pos]) < 1:    # 잔다 VV+EC 등을 분해함
    #                 decomposed_morpheme = morpheme[:]
    #
    #             # 문법 형태소이면
    #             # elif pos in grammatical_pos:  # 잔다 VV+EC 등을 분해하지 않음
    #             elif sum([1 for pos in pos.split("+") if pos in self.grammatical_pos]) >= 1: # 잔다 VV+EC 등을 분해함
    #                 decomposed_morpheme = "".join([self.transform_v2(char) if character_is_korean(char) else char for char in morpheme])
    #
    #             new_eojeol.append(decomposed_morpheme)
    #
    #         if morpheme_tokenization == False:  # 형태소 토큰화 없이 어절 그대로 자모로 변환만 한다면
    #             if flatten == True:
    #                 new_sent.append("".join(new_eojeol))
    #             elif flatten == False:
    #                 new_sent.append(new_eojeol)
    #         elif morpheme_tokenization == True: # 형태소 토큰화 + 자모 변환 한다면
    #             if flatten == True:
    #                 new_sent += new_eojeol
    #             elif flatten == False:
    #                 new_sent.append(new_eojeol)
    #
    #     if flatten == True:
    #         new_sent = doublespace_pattern.sub(" ", " ".join(new_sent))
    #     elif flatten == False:
    #         pass
    #
    #     return new_sent
    #
    #
    # ## decomposed_morphological용
    # # 형태소 분석 과정이 있기 때문에 조금씩 노이즈 발생
    # def str2jamo_morphological_with_space_symbol(self, sent, use_original):
    #     '''
    #     :param sent: 자모 변환할 문장      '너를 좋아해'
    #     :param morpheme_analysis:
    #         False: 자모 변환만 수행    (어절 토큰화 문장을 자모로 변환하는 데에 그대로 이용 가능)
    #         True: 형태소 분석 + 자모 변환
    #     :param use_original: konlpy original mecab 쓸지
    #     :return: 자모 변환된 문장          '너ㅡㄹ 좋아해' or '너 ㄹㅡㄹ 좋아해'
    #     '''
    #
    #     if use_original == True:
    #         mors_ejs_in_sent = self.mc_orig.pos(sent, flatten=False)  # 형태소 분석
    #     elif use_original == False:
    #         mors_ejs_in_sent = self.mc_fixed.pos(sent, flatten=False)  # 형태소 분석
    #
    #     new_sent = list()
    #     for ix in range(len(mors_ejs_in_sent)):
    #         eojeol = mors_ejs_in_sent[ix]  # [('나', 'NP'), ('는', 'JX')]
    #
    #         new_eojeol = list()  # ['나', 'ㄴㅡㄴ']
    #         for jx in range(len(eojeol)):
    #             morpheme, pos = eojeol[jx]  # '너', 'NP'
    #
    #             # 문법 형태소가 아니면
    #             # if not pos in grammatical_pos:    # 잔다 VV+EC 등을 분해하지 않음
    #             if sum([1 for pos in pos.split("+") if pos in self.grammatical_pos]) < 1:  # 잔다 VV+EC 등을 분해함
    #                 decomposed_morpheme = morpheme[:]
    #
    #             # 문법 형태소이면
    #             # elif pos in grammatical_pos:  # 잔다 VV+EC 등을 분해하지 않음
    #             elif sum([1 for pos in pos.split("+") if pos in self.grammatical_pos]) >= 1:  # 잔다 VV+EC 등을 분해함
    #                 decomposed_morpheme = "".join(
    #                     [self.transform_v2(char) if character_is_korean(char) else char for char in morpheme])
    #
    #             new_eojeol.append(decomposed_morpheme)
    #
    #         new_sent.append(new_eojeol)
    #
    #         # if morpheme_tokenization == False:  # 형태소 토큰화 없이 어절 그대로 자모로 변환만 한다면
    #         #     # if flatten == True:
    #         #     #     new_sent.append("".join(new_eojeol))
    #         #     # elif flatten == False:
    #         #     new_sent.append(new_eojeol)
    #         # elif morpheme_tokenization == True:  # 형태소 토큰화 + 자모 변환 한다면
    #         #     # if flatten == True:
    #         #     #     new_sent += new_eojeol
    #         #     # elif flatten == False:
    #         #     new_sent.append(new_eojeol)
    #
    #     # if flatten == True:
    #     #     new_sent = doublespace_pattern.sub(" ", " ".join(new_sent))
    #     # elif flatten == False:
    #     #     pass
    #
    #
    #     new_sent_with_special_token = list (chain.from_iterable( self.intersperse(new_sent, self.space_symbol) ) )
    #
    #     return new_sent_with_special_token


    # lexical grammatical 토크나이저
    # 어절에서 조사와 어미만 분리하기 # 한국초등학교에서는 -> 육식동물/LEXICAL, '에서/JKB', '는/JX'
    def mecab_grammatical_tokenizer(self, sent: str, nfd: bool):

        if nfd == False:
            mors_ejs_in_sent = self.mc_fixed.pos(sent, flatten=False, coda_normalization=True)  # 형태소 분석
        elif nfd == True:
            mors_ejs_in_sent = self.mc_fixed.pos(sent, flatten=False, coda_normalization=False)  # 형태소 분석


        # def get_grammatical_idx(eojeol: list):
        #     idx = len(eojeol) - 1
        #     while idx >= 0:
        #         if (eojeol[idx][-1] in self.grammatical_pos) or (eojeol[idx][-1] in self.punctuation_pos):
        #             idx -= 1
        #         else:
        #             break
        #     return idx + 1  # 어절 내에서 가장 먼저 나온 문법 형태소 인덱스    # 한국대학교'에서'는
        #
        # new_sent = list()
        #
        # for ix in range(len(mors_ejs_in_sent)):
        #     eojeol = mors_ejs_in_sent[ix]
        #
        #     idx_grammatical_start = get_grammatical_idx(eojeol=eojeol)
        #
        #     lexical_part = [("".join([mor_pos[0] for mor_pos in eojeol[:idx_grammatical_start]]), "LEXICAL")]   # ('한국대학교', 'LEXICAL')
        #     grammatical_part = eojeol[idx_grammatical_start:]   # [('에서', 'JKB'), ('는', 'JX')]
        #     grammatical_tokenized_eojeol = lexical_part + grammatical_part    # [('한국대학교', 'LEXICAL'), ('에서', 'JKB'), ('는', 'JX')]
        #
        #     new_sent.append(grammatical_tokenized_eojeol)


        # 어휘 형태소 이어 붙이기
        # '이뻐보였다' 디버깅
        def concat_lexical_morphemes(eojeol: list):
            lexical_start_idxs = list()
            # grammatical_start_idxs = list()
            lexical_end_idxs =list()

            for idx, cnt in enumerate(eojeol):
                if idx != 0:
                    if (cnt[-1] in self.lexical_pos) and (not eojeol[idx - 1][-1] in self.lexical_pos):  # 어휘 형태소이고, 이전 형태소가 어휘 형태소가 아니라면(문법 형태소 or 기호라면)
                        lexical_start_idxs.append(idx)
                    # elif (not cnt[-1] in self.lexical_pos) and (eojeol[idx - 1][-1] in self.lexical_pos):  # 어휘 형태소가 아니고(문법 형태소 or 기호이고), 이전 형태소가 어휘 형태소이면
                    #     grammatical_start_idxs.append(idx)
                    elif (not cnt[-1] in self.lexical_pos) and (eojeol[idx - 1][-1] in self.lexical_pos):  # 어휘 형태소가 아니고(문법 형태소 or 기호이고), 이전 형태소가 어휘 형태소이면
                        lexical_end_idxs.append(idx)

                    if idx == len(eojeol) - 1:  # 어절 마지막 형태소이면

                        # if (len(lexical_start_idxs) >= 1) and lexical_start_idxs[-1] < idx + 1:    # lexical_start_idx가 하나라도 있고, 현재 인덱스+1보다 이전에 lexical_start_idx가 있으면
                        if (len(lexical_start_idxs)-1 == len(lexical_end_idxs)) and lexical_start_idxs[-1] < idx + 1:    # lexical_start_idx가 lexical_end_idx보다 하나 많고, 현재 인덱스+1보다 이전에 lexical_start_idx가 있으면
                            lexical_end_idxs.append(idx+1)

                elif idx == 0:
                    # if (not cnt[-1] in self.grammatical_pos):  # 어휘 형태소이면
                    if (cnt[-1] in self.lexical_pos):  # 어휘 형태소이면
                        lexical_start_idxs.append(idx)
                    # elif cnt[-1] in self.grammatical_pos:  # 어휘 형태소가 아니면(문법 형태소 or 기호이면)
                    #     grammatical_start_idxs.append(idx)
                        if len(eojeol)==1:  # 어휘 형태소 하나로만 된 어절이면
                            lexical_end_idxs.append(idx+1)  # end_idx까지 추가


            assert(len(lexical_start_idxs)==len(lexical_end_idxs)), ("concat_lexical_morphemes ERROR!!!")    # 무결성 검사

            lexical_idxs = list(zip(lexical_start_idxs, lexical_end_idxs))    # 어절 내 어휘 형태소의 (시작, 끝) 인덱스의 list

            # return lexical_idxs

            new_eojeol = eojeol[:]

            for jx in range(len(lexical_idxs)):
                concat_eojeol_proto = [ "DUMMY" ] * (lexical_idxs[jx][-1]-lexical_idxs[jx][0])  # 띄어쓰기 안 돼서 한 어절 내에 LEXICAL 여러 번 나타나는 경우 대비해, 인덱스 유지하기 위한 DUMMY 형태소 추가
                # concat_eojeol_proto = [ ( "".join( [mor_pos[0] for mor_pos in eojeol[lexical_idxs[jx][0]:lexical_idxs[jx][-1]] ] ), "LEXICAL" )  ]
                concat_eojeol_proto[0] = ( "".join( [mor_pos[0] for mor_pos in eojeol[lexical_idxs[jx][0]:lexical_idxs[jx][-1]] ] ), "LEXICAL" )

                # [ ( "".join( [mor_pos[0] for mor_pos in eojeol[lexical_idxs[jx][0]:lexical_idxs[jx][-1]] ] ), "LEXICAL" )*(lexical_idxs[jx][-1]-lexical_idxs[jx][0]) ]

                new_eojeol[lexical_idxs[jx][0]:lexical_idxs[jx][-1]] = concat_eojeol_proto

            new_eojeol = [mor_pos for mor_pos in new_eojeol if mor_pos != "DUMMY"]  # DUMMY 형태소 제거

            return new_eojeol



            # new_eojeol = eojeol[:]
            #
            #
            # new_eojeol[lexical_idxs[0][0]:lexical_idxs[0][-1]] = ( "".join( [mor_pos[0] for mor_pos in eojeol[lexical_idxs[0][0]:lexical_idxs[0][-1]] ] ), "LEXICAL" )

        # def concat_lexicals(eojeol: list, lexical_idxs: tuple):
        #
        #     for jx in range(len(lexical_idxs)):
        #         eojeol[lexical_idxs[jx][0]:lexical_idxs[jx][-1]] = ( "".join( [mor_pos[0] for mor_pos in eojeol[lexical_idxs[jx][0]:lexical_idxs[jx][-1]] ] ), "LEXICAL" )
        #
        #     return eojeol




        new_sent = list()

        for ix in range(len(mors_ejs_in_sent)):
            eojeol = mors_ejs_in_sent[ix]

            new_eojeol = concat_lexical_morphemes(eojeol=eojeol)

            new_sent.append(new_eojeol)


        return new_sent

        # if flatten == True:
        #     new_sent_with_special_token = list(chain.from_iterable(self.intersperse(new_sent, self.space_symbol)))
        #
        #     if self.space_symbol == "": # 스페이스 심벌 안 쓴다면
        #         new_sent_with_special_token = [token for token in new_sent_with_special_token if token != ""]    # 빈 토큰 제외

        # elif flatten == False:

        # new_sent_with_special_token = self.intersperse(new_sent, self.space_symbol)  # [['너', 'ㄴ'], ['▃'], ['날'], ['▃'], ['좋', '아', '하', '아']]
        #
        # if self.space_symbol == "":  # 스페이스 심벌 안 쓴다면
        #     new_sent_with_special_token = [token for token in new_sent_with_special_token if token != [""]]  # 빈 토큰 제외
        #
        # return new_sent_with_special_token




## 자모 분해된 것을 원래 문장으로 복원. 완벽하지 않음.
    def jamo2str_morphological(self, jamo):
        jamo_eojeols = jamo.split(" ")  # ['나는', '즐겁게', '밥을', '먹는다.']

        # 각 토큰을 음절 단위로 분해
        eojeols_not_composed = list()   # [['나', 'ㄴㅡㄴ'], ['즐', '겁', 'ㄱㅔ-'], ['밥', 'ㅇㅡㄹ'], ['먹', 'ㄴㅡㄴ', 'ㄷㅏ-', '.']]

        for ix in range(len(jamo_eojeols)):
            jamo_eojeol = jamo_eojeols[ix]      # '나ㄴㅡㄴ'

            composed_eojeol, idx = list(), 0    # ['나', 'ㄴㅡㄴ']
            while idx < len(jamo_eojeol):
            # for jx in range(len(jamo_eojeol)):
            #     jamo_eojeol[idx]

                if character_is_complete_korean(jamo_eojeol[idx]):  # 이미 합쳐져 있는 음절 문자이면
                    composed_eojeol.append(jamo_eojeol[idx])
                    idx += 1

                elif character_is_jaum(jamo_eojeol[idx]):   # 분해된 자모이면
                    composed_eojeol.append(jamo_eojeol[idx:idx + 3])
                    idx += 3

                    # if jamo_eojeol[idx:idx + 3].count(dummy_letter) != 2:    # 일반적인 경우 (ㄱㅏㄴ, ㄱㅏ-)
                    #     composed_eojeol.append(jamo_eojeol[idx:idx + 3])
                    #     idx += 3
                    # elif jamo_eojeol[idx:idx + 3].count(dummy_letter) != 2:   # 자음으로만 된 형태소이면(ㄴ, ㄹ, ..)
                    #     if ix

                else:   # 기타: 특수 기호 등
                    composed_eojeol.append(jamo_eojeol[idx])
                    idx += 1

            eojeols_not_composed.append(composed_eojeol)


        composed_str = list()   # ['나는', '괜찮아']
        for jx in range(len(eojeols_not_composed)):
            eojeol_not_composed = eojeols_not_composed[jx]  # ['나', 'ㄴㅡㄴ'],

            eojeol_composed = str()
            for kx in range(len(eojeol_not_composed)):
                char_not_composed = eojeol_not_composed[kx]

                if len(char_not_composed) == 3 and sum([1 for char in char_not_composed if character_is_jaum(char)]) >= 1:  # 합쳐야 되는 문자열이면
                    if not self.dummy_letter in char_not_composed:
                        eojeol_composed += compose(char_not_composed[0], char_not_composed[1], char_not_composed[2])
                    elif self.dummy_letter in char_not_composed:
                        if char_not_composed.count(self.dummy_letter) == 1:   # ㅅㅏ- 등의 일반적인 경우
                            eojeol_composed += compose(char_not_composed[0], char_not_composed[1], " ")
                        elif char_not_composed.count(self.dummy_letter) == 2:   # ㄴ-- (조사 'ㄴ') 등의 경우
                            # eojeol_composed += compose(char_not_composed[0], " ", " ")

                            if kx != 0: # 앞 음절이 있으면
                                chosung, junsung, jonsung = list(decompose(eojeol_not_composed[kx-1])[:2]) + [eojeol_not_composed[kx].replace(self.dummy_letter, "")] # 앞 음절 분해 후 붙이기
                                eojeol_composed = eojeol_composed[:kx-1]    # 앞 음절을 eojeol_composed에서 삭제
                                eojeol_composed += compose(chosung, junsung, jonsung)  # 앞 음절에 붙여서 합성   나 + ㄴ -> 난
                            elif kx == 0:   # 앞 음절이 없으면
                                eojeol_composed += char_not_composed

                else:   # 합치지 말아야 될 문자열이면 (이미 합쳐져 있는 음절 문자, 특수 기호 등)
                    eojeol_composed += char_not_composed

            composed_str.append(eojeol_composed)


        # 합성 안 된 것이 있는지 체크
        not_composed_idx =  [ix for ix, token in enumerate(composed_str) if self.dummy_letter in token]
        if len(not_composed_idx) >=1 :  # ['나', 'ㄴ--', '괜찮', '아'] 같은 경우
            for lx in range(len(not_composed_idx)):
                if not_composed_idx[lx] >= 1:
                    previous_token = composed_str[not_composed_idx[lx]-1] # 바로 앞 토큰

                    if decompose(previous_token[-1])[-1] == ' ':    # 앞 음절(바로 앞 토큰의 마지막 음절)이 종성이 없다면
                        chosung, junsung, jonsung = list(decompose(previous_token[-1])[:2]) + [composed_str[ not_composed_idx[lx] ].replace(self.dummy_letter, "")]

                        new_previous_token = self.str_substitution(orig_str=previous_token, sub_idx=-1, sub_str=compose(chosung, junsung, jonsung))
                        composed_str[not_composed_idx[lx]-1] = new_previous_token[:]    # 앞 토큰을 새로 합성한 것으로 치환   # '나' > '난'
                        composed_str[not_composed_idx[lx]] = ""  # 현재 토큰은 공백으로 치환

                    elif decompose(previous_token[-1])[-1] != " ":    # 앞 음절(바로 앞 토큰의 마지막 음절)이 종성이 있다면
                        composed_str[not_composed_idx[lx]] = composed_str[not_composed_idx[lx]].replace(self.dummy_letter, "")    # 그냥 coda letter만 삭제하기

                elif not_composed_idx[lx] == 0:
                        composed_str[not_composed_idx[lx]] = composed_str[not_composed_idx[lx]].replace(self.dummy_letter, "")    # 그냥 coda letter만 삭제하기

        composed_str = [token for token in composed_str if token != ""]

        return " ".join(composed_str)   # "나는 괜찮아"












############################################################################################################################################################################################



# dummy_letter = "⊸"  # chr(8888)
# space_symbol = "▃"  # chr(9603)
# grammatical_symbol = "⭧"  # chr(11111)
#
#
#
# tok = tokenizers(dummy_letter="#", space_symbol="▃", grammatical_symbol=["", ""], nfd=True)    #
# tok = tokenizers(dummy_letter="#", space_symbol="", grammatical_symbol=["", ""], nfd=False)
# tok = tokenizers(dummy_letter="", space_symbol="", grammatical_symbol=["⫸", "⭧"], nfd=False)
#
# tok = tokenizers(dummy_letter="", space_symbol="", grammatical_symbol=["⫸", "⭧"], nfd=True)
# tok = tokenizers(dummy_letter="", space_symbol="", grammatical_symbol=["⫸", "⭧"], nfd=False)
# tok = tokenizers(dummy_letter="", space_symbol="", grammatical_symbol=[chr(7777), chr(9999)], nfd=False)
# self = tok
#
# sent = "이것이 아니다"
# sent = "재밌음ㅋㅋ"
# sent = "재밌음ㅠㅠ"
# sent = "넌 날 좋아해"
# sent = "미궁에서 뜬 아앗"
# sent = "훌륭한 사망 플래그의 예시이다"
# sent = "수해에 입장한다"   # ['ㅅㅜ#ㅎㅐ#', 'ㅇㅔ#', '▃', 'ㅇㅣㅂㅈㅏㅇ', 'ㅎㅏ#', 'ㄴ##ㄷㅏ#']
# sent = "나는 널 좋아해"
# sent = "발생하는 한국고려대학교에서는 빨리는 먹지는 못해서요."
# sent = "그렇지만은요"
# sent = "인정되었어요"
# sent = "계속되었어요"
# sent = "대응하는"
# sent = "합법적이다"
# sent = "한국고려대학교에서는일본도쿄대학교를빨리는 먹지는 못해서요."    # lexical_grammatical로 알아서 띄어쓰기. 조사, 어미이면 띄어 씀.
#
# tok.str2jamo(sent)   # 'ㄴㅓㄴ ㄴㅏㄹ ㅈㅗㅎㅇㅏ#ㅎㅐ#'
# tok.jamo2str(tok.str2jamo(sent))
#
# tok2.str2jamo(sent)   # 'ㄴㅓㄴ ㄴㅏㄹ ㅈㅗㅎㅇㅏㅎㅐ'
# # tok2.jamo2str(tok2.str2jamo(sent))  # '넌 날 좋앟ㅐ'
# moasseugi(tok2.str2jamo(sent))  # '넌 날 좋아해'
# # moasseugi('들어가ㄴ다')
#
#
# sent = "예쁜 가방"
# sent = "난 너를 좋아해."
# sent = "전통으로 등장하는 대사"
#
# sent = "지하철에서 완전 가깝고 주변에 마트도 가까워요."
# sent = "호스팅이 필요한지 여부로 판단한다"
# sent = "뭔지 모르지만"
# sent = "이듬해, 견심은 주공소공의 고사를 본받아 평안을 좌백에, 견풍을 우백에 임명해야 한다는 내용의 부명을 지어 바쳤고, 왕망은 곧바로 부명대로 하였다. 그런데 견풍이 정식으로 취임하기 직전에 견심이 부명을 또 지어 바쳤는데, 내용은 이러하였다."
# sent = "사이몬페그의 익살스런 연기가 돋보였던 영화!스파이더맨에서 늙어보이기만 했던 커스틴 던스트가 너무나도 이뻐보였다"
# sent = "도자이선 하행 난고7쵸메 난고18쵸메 오야치 신삿포로 방면"
# sent = "이 효과로 카드의 발동을 무효로 할 때마다, 이 카드의 공격력과 수비력은 500 포인트 내린다."
#
#
# # eojeol
# ee = tok.mecab_tokenizer(sent, token_type="eojeol", tokenizer_type="none", decomposition_type="composed", flatten=False); print(ee)
# ee = tok.mecab_tokenizer(sent, token_type="eojeol", tokenizer_type="none", decomposition_type="decomposed_pure"); print(ee)
# # ee = tok.mecab_tokenizer(sent, token_type="eojeol", tokenizer_type="mecab_fixed", decomposition_type="decomposed_lexical"); print(ee)
# # ee = tok.mecab_tokenizer(sent, token_type="eojeol", tokenizer_type="mecab_fixed", decomposition_type="decomposed_grammatical"); print(ee)
#
# # morpheme
# ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="composed", flatten=True); print(ee)
# ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="composed", flatten=False); print(ee)
# # ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="composed_nfd"); print(ee)
# ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="decomposed_pure", flatten=True); print(ee)
# ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="decomposed_pure", flatten=False); print(ee)
# ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="decomposed_lexical"); print(ee)
# ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="decomposed_lexical", flatten=False); print(ee)
# ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="decomposed_grammatical"); print(ee)
# ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_orig", decomposition_type="decomposed_grammatical", flatten=False); print(ee)
#
# ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="composed"); print(ee)
# ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="composed", flatten=False); print(ee)
# ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure"); print(ee)
# ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure", flatten=False); print(ee)
# ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="decomposed_lexical"); print(ee)
# ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="decomposed_grammatical"); print(ee)
#
# ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="composed", lexical_grammatical=True); print(ee)
# ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="composed", lexical_grammatical=True, flatten=False); print(ee)
# ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure", lexical_grammatical=True); print(ee)
# ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure", lexical_grammatical=True, flatten=False); print(ee)
# ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="decomposed_lexical", lexical_grammatical=True); print(ee)
# ee = tok.mecab_tokenizer(sent, token_type="morpheme", tokenizer_type="mecab_fixed", decomposition_type="decomposed_grammatical", lexical_grammatical=True,); print(ee)
#
#
#
# len(ee[0])
# len(ee[1])
# len(ee[2])
# len(ee[3])
# len(ee[4])
# len(ee[5])
#
# # mecab original
# #     composed
# tok.mecab_tokenizer(sent, tokenizer_type="mecab_orig", decomposition_type="decomposed_morphological") # ['넌', '▃', '날', '▃', '좋', '아', '해']
# ee = tok.mecab_tokenizer(sent, tokenizer_type="mecab_orig", decomposition_type="decomposed_morphological_nfd") # ['넌', '▃', '날', '▃', '좋', '아', '해']
# ee = tok.mecab_tokenizer(sent, tokenizer_type="mecab_orig", decomposition_type="decomposed_pure_nfd") # ['넌', '▃', '날', '▃', '좋', '아', '해']
#
# ee = tok.mecab_tokenizer(sent, tokenizer_type="mecab_fixed", decomposition_type="decomposed_morphological_nfd") # ['넌', '▃', '날', '▃', '좋', '아', '해']
#
# ee = tok.mecab_tokenizer(sent, tokenizer_type="mecab_fixed", decomposition_type="decomposed_pure_nfd") # ['넌', '▃', '날', '▃', '좋', '아', '해']
#
# len(ee[0])
# len(ee[1])
# len(ee[2])
# len(ee[3])
# len(ee[4])
# len(ee[5])
# len(ee[6])
# len(ee[7])
#
#     # decomposed pure
# tok.mecab_tokenizer(sent, use_original=True, pure_decomposition=True)  # ['ㄴㅓㄴ', '▃', 'ㄴㅏㄹ', '▃', 'ㅈㅗㅎ', 'ㅇㅏ#', 'ㅎㅐ#']
# tok2.mecab_tokenizer(sent, use_original=True, pure_decomposition=True)  # ['ㄴㅓㄴ', '▃', 'ㄴㅏㄹ', '▃', 'ㅈㅗㅎ', 'ㅇㅏ', 'ㅎㅐ']
#     # decomposed morphological
# tok.mecab_with_morphological_decomposition(sent, use_original=True)  # ['ㄴㅓㄴ', '▃', '날', '▃', '좋', 'ㅇㅏ#', 'ㅎㅐ#']
# tok2.mecab_with_morphological_decomposition(sent, use_original=True)  # ['ㄴㅓㄴ', '▃', '날', '▃', '좋', 'ㅇㅏ', 'ㅎㅐ']
#
# # mecab fixed
#     # composed
# tok.mecab_tokenizer(sent, use_original=False, pure_decomposition=False) # ['너', 'ㄴ', '▃', '날', '▃', '좋', '아', '하', '아']
#     # decomposed pure
# tok.mecab_tokenizer(sent, use_original=False, pure_decomposition=True)  # ['ㄴㅓ#', 'ㄴ##', '▃', 'ㄴㅏㄹ', '▃', 'ㅈㅗㅎ', 'ㅇㅏ#', 'ㅎㅏ#', 'ㅇㅏ#']
# tok2.mecab_tokenizer(sent, use_original=False, pure_decomposition=True)  # ['ㄴㅓ', 'ㄴ', '▃', 'ㄴㅏㄹ', '▃', 'ㅈㅗㅎ', 'ㅇㅏ', 'ㅎㅏ', 'ㅇㅏ']
#     # decomposed morphological
# tok.mecab_with_morphological_decomposition(sent, use_original=False)  # ['너', '##ㄴ', '▃', '날', '▃', '좋', 'ㅇㅏ#', '하', 'ㅇㅏ#']
# tok2.mecab_with_morphological_decomposition(sent, use_original=False)  # ['너', 'ㄴ', '▃', '날', '▃', '좋', 'ㅇㅏ', '하', 'ㅇㅏ']
#
#
# # 자음 문법 형태소 처리: ##ㄴ
# # 원래 종성 위치대로.
# #
# # 그냥 자음/모음 not 문법 형태소 처리: ㅋ##, #ㅠ#
# # 자음은 초성, 모음은 중성 처리.
#
#
#
#
# mc = Mecab(use_original=False)
# sent = "미궁에서 뜬 아앗"
# mc.pos(sent, join=False)
# mc.pos(sent, join=True)
