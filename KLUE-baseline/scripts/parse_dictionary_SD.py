# 표준국어대사전의 정보를 하나의 dataframe으로 압축
import os
import re
import pickle
import pandas as pd


# load as a dataframe   # 만들어진 것 불러와서 유지 보수하기 위함
# SDK_all_sorted = pd.read_pickle("./output/parsed_dic_linux_orig.pkl")
# SDK_all_sorted = pd.read_pickle("./output/parsed_dic_linux_fixed.pkl")


# Parsing: Standard Korean Dic.
# path_SKD = "./SJ/전체 내려받기_표준국어대사전_xls_20210107/"
path_SKD = "../전체 내려받기_표준국어대사전_xls_20211101/"
list_SKD = sorted(os.listdir(path_SKD))
list_SKD = [file for file in list_SKD if file.endswith(".xls")]

SDK_all = pd.DataFrame()

for j in range(len(list_SKD)):
    cont_SKD = pd.read_excel(path_SKD + list_SKD[j])
    SDK_all = SDK_all.append(cont_SKD)

SDK_all_sorted = SDK_all.sort_values(by="어휘", ascending=True)   # ㄱㄴㄷ sorting
SDK_all_sorted = SDK_all_sorted.reset_index(drop=True)

    # insert a new column for homonym
SDK_all_sorted.insert(1, "num", value=0)
# SDK_all_sorted["num"] = [0 for ix in range(len(SDK_all)) if re.search(regex_word)]

    # insert a new column for 어간
SDK_all_sorted.insert(2, "어간_only_hanguel", value="")
SDK_all_sorted.insert(2, "어간", value="")

    # insert a new column for 어휘_no_homonym    (초-(22) > 초-, 체화(01) > 체화)   # (숫자) 부분만 제거
SDK_all_sorted.insert(2, "어휘_no_homonym", value="")
p_num = re.compile("\(\d+\)")
SDK_all_sorted["어휘_no_homonym"] = [p_num.sub("", x) for x in SDK_all_sorted["어휘"]]


    # insert a new column for 어휘_orig (쌤(03) > 쌤, 쌩그레-하다 > 쌩그레하다)   # 한글 이외의 문자 다 제거
SDK_all_sorted.insert(2, "어휘_only_hanguel", value="")
p_not_hanguel = re.compile("[^ㄱ-ㅣ가-힣]")  # https://kynk94.github.io/devlog/post/re-match-hangul
SDK_all_sorted["어휘_only_hanguel"] = [p_not_hanguel.sub("", x) for x in SDK_all_sorted["어휘"]]


    # trim the POS column
p_POS = re.compile("「(.+)」")    # regex for POS
SDK_all_sorted["품사"] =  ["\t".join(p_POS.findall(x)) for x in SDK_all_sorted["품사"]]   # from '[Ⅰ]「어미」\n[Ⅱ]「품사 없음」\n'




# 어미, 조사만 추출
em_idx = [ix for ix in range(len(SDK_all_sorted)) if ("어미" in SDK_all_sorted.loc[ix, "품사"])]
js_idx = [ix for ix in range(len(SDK_all_sorted)) if ("조사" in SDK_all_sorted.loc[ix, "품사"])]

em_df = SDK_all_sorted.loc[em_idx,:]
js_df = SDK_all_sorted.loc[js_idx,:]

em_df.drop(['어휘_only_hanguel', '어휘_no_homonym' , '어간', '어간_only_hanguel', '구성 단위', '고유어 여부', '원어', '어원',
            '주표제어', '부표제어', '발음', '활용', '검색용 이형태', '공통 문형', '의미 문형', '뜻풀이', '전문 분야', '속담', '관용구',
            '대역어', '생물 분류군 정보', '멀티미디어'])

em_df.to_excel("../표준_어미.xlsx")
em_df.to_csv("../표준_어미.csv")
js_df.to_excel("../표준_조사.xlsx")
js_df.to_csv("../표준_조사.csv")



    # 어간 추출 (형태소 분석된 고유어 어간 찾기 위해. 느끼다 > 느끼)
# SDK_all_sorted["어간"] = [re.sub("다$", "", SDK_all_sorted.iloc[ix, 2]) for ix in range(len(SDK_all_sorted)) if ("동사" in SDK_all_sorted.iloc[ix, 13].split("\t")) or ("형용사" in SDK_all_sorted.iloc[ix, 13].split("\t"))]
for ix in range(len(SDK_all_sorted)):
    if ix % 1000 == 0: print(ix)

    # 어간 추출 (형태소 분석된 고유어 어간 찾기 위해. 느끼다 > 느끼)
    if ("동사" in SDK_all_sorted.iloc[ix, 14].split("\t")) or ("형용사" in SDK_all_sorted.iloc[ix, 14].split("\t")): # 해당 형태소가 용언이면
        SDK_all_sorted.iloc[ix, 4] = re.sub("다$", "", SDK_all_sorted.iloc[ix, 3])
                        # 어간 열

# # 접미사 어간 수동 입력
# # 하/XSV, 시키/XSV, 되/XSV, 히/XSV, 당하/XSV, 받/XSV, 거리/XSV  # 동사 파생
# # 스럽/XSA, 되/XSA, 롭/XSA, 하/XSA, 같/XSA, 답/XSA   # 형용사 파생
#
# SDK_all_sorted[SDK_all_sorted["어휘"]=="-답다"]
# SDK_all_sorted.iloc[446, 4] = '-답'

    # only 한글 어간 행 생성
SDK_all_sorted["어간_only_hanguel"] = [p_hanguel.sub("", x) for x in SDK_all_sorted["어간"]]



    # regex for homonym numbering   서신(01) > 01
regex_num_paren = re.compile("\((\d+)\)")  # (01)
# regex_num_only = re.compile("\d+")   # 01

SDK_all_sorted.iloc["num"] = [regex_num_paren.search(SDK_all_sorted.iloc[ix, 0]).group(1) for ix in range(len(SDK_all_sorted)) if re.search(regex_num_paren, SDK_all_sorted.iloc[ix, 0])]

for ix in range(len(SDK_all_sorted)):
    if ix % 1000 == 0: print(ix)

    if re.search(regex_num_paren, SDK_all_sorted.iloc[ix, 0]):  # If there is a number string in a word
        SDK_all_sorted.iloc[ix, 1] = regex_num_paren.search(SDK_all_sorted.iloc[ix, 0]).group(1)    # 01
        # SDK_all_sorted.iloc[ix, 0] = regex_num_paren.sub("", SDK_all_sorted.iloc[ix, 0])    # 서신


# add a column for POS information
# 한자어와 관련된 것만 처리하도록 함. 조사 어미 등 무시.
def to_POS(dic_row):
    Pumsas = dic_row["품사"].split("\t")  # POS info. of 표준국어대사전
    POSs = []   # an empty list for saving POSs

    for ix in range(len(Pumsas)):
        if Pumsas[ix] == "명사":
            POSs.append("NNG")  # 고유명사(NNP)도 일반명사(NNG)로 처리함. 코퍼스는 둘 구분하나, 사전은 구분 안 함. 이래서 문제 발생함. '일본'이 대표적.
        elif Pumsas[ix] == "의존 명사":
            POSs.append("NNB")
        elif Pumsas[ix] == "대명사":
            POSs.append("NP")
        elif Pumsas[ix] == "수사":
            POSs.append("NR")
        elif Pumsas[ix] == "동사":
            POSs.append("VV")
        elif Pumsas[ix] == "형용사":
            POSs.append("VA")
        elif Pumsas[ix] in ["보조 동사", "보조 형용사"]:
            POSs.append("VX")
        elif Pumsas[ix] == "관형사":
            POSs.append("MM")
        elif Pumsas[ix] == "부사":
            POSs.append("MAG")
            # 접속 부사(MAJ)는 '표준'에서 별도의 품사로 규정하고 있지 않아 무시. 어차피 한자어도 없을 듯.
        elif Pumsas[ix] == "감탄사":
            POSs.append("IC")

        # elif Pumsas[ix] == "접사":    # 표제어의 하이픈이 어디에 있는지에 따라 구별
        #     if (dic_row['어휘'].startswith("-")) and (not dic_row['어휘'].endswith("-")): # 앞에만 - 있으면 (-개)
        #         POSs.append("XS*")  # 접미사 XSN, XSV, XSA 중 하나
        #     elif (not dic_row['어휘'].startswith("-")) and (dic_row['어휘'].endswith("-")):  # 뒤에만 - 있으면  (최-)
        #         POSs.append("XPN")  # 체언 접두사
        #     elif (dic_row['어휘'].startswith("-")) and (dic_row['어휘'].endswith("-")):   # 앞뒤 모두 - 있으면 (-기-)
        #         POSs.append("XS*")  # 접미사 XSN, XSV, XSA 중 하나

        # elif Pumsas[ix] == "어미":    # 표제어의 하이픈이 어디에 있는지에 따라 구별
        #     if (dic_row['어휘'].startswith("-")) and (not dic_row['어휘'].endswith("-")): # 앞에만 - 있으면 (-나이다)
        #         POSs.append("EF/EC")
    return POSs

column_POS = [to_POS(SDK_all_sorted.iloc[ix]) for ix in range(len(SDK_all_sorted))]
SDK_all_sorted["POS"] = ""  # 초기화
SDK_all_sorted["POS"] = column_POS[:]

    # 사전 내 모든 품사
# xx = set(SDK_all_sorted["품사"])
# gg = [x.split("\t") for x in xx]
# flat_list = [item for sublist in gg for item in sublist]
# set(flat_list)
#
# 명사 (일반 명사)
# 의존 명사
# 대명사
# 수사
#
# 동사
# 형용사
# 보조 동사
# 보조 형용사
#
# 관형사
#
# 부사
#
# 감탄사
#
# 접사
#
# 어미, 조사, 품사 없음, 구



### 수동으로 품사 정보 입력
# xpn
# ['맏/XPN', '가/XPN', '반/XPN', '제/XPN', '피/XPN', '생/XPN', '초/XPN', '맨/XPN', '고/XPN',
# '노/XPN', '탈/XPN', '반_x07/XPN', '과/XPN', '날/XPN', '헷/XPN', '부/XPN', '역/XPN', '불/XPN', '미/XPN',
# '풋/XPN', '대/XPN', '폐/XPN', '재/XPN', '최/XPN', '저/XPN', '범/XPN', '비/XPN', '다/XPN', '정/XPN',
# '신/XPN', '주/XPN', '구/XPN', '한/XPN', '무/XPN', '왕/XPN', '소/XPN', '개/XPN', '헛/XPN', '준/XPN', '친/XPN']
# xsn
# ['째/XSN', '형/XSN', '들이/XSN', '찌리/XSN', '논/XSN', '꾼/XSN', '층/XSN', '급/XSN', '박이/XSN', '변/XSN',
# '쟁이/XSN', '기/XSN', '적/XSN', '론/XSN', '업/XSN', '대__21/XSN', '률/XSN', '분/XSN', '장이/XSN', '풍/XSN',
# '투성이/XSN', '씩/XSN', '네/XSN', '생/XSN', '광/XSN', '권/XSN/NNG', '당/XSN', '소/XSN', '덜/XSN', '끼리/XSN',
# '대/XSN', '깨/XSN', '권/XSN', '꼴/XSN', '율/XSN', '가/XSN', '댁/XSN', '판/XSN', '무/XSN', '__01/XSN',
# '류/XSN', '치레/XSN', '떨/XSN', '치/XSN', '저/XSN', '질/XSN', '행/XSN', '별/XSN', '상/XSN', '군/XSN',
# '화/XSN/NNG', '계/XSN', '직/XSN', '재/XSN', '이/XSN', '들/XSN', '제/XSN', '설/XSN', '끼/XSN', '여/XSN',
# '성/XSN', '님/XSN', '께/XSN', '바기/XSN', '배기/XSN', '용/XSN', '산/XSN', '리/XSN', '시/XSN', '짜리/XSN',
# '쯤/XSN', '뻘/XSN', '화/XSN', '경/XSN', '질__/XSN', '니임/XSN', '구/XSN']
# xsa
# ['만하/XSA', '스럽/XSA', '되/XSA', '시럽/XSA', '롭/XSA', '스러/XSA', '하/XSA', '같/XSA', '허/XSA',
# '만허/XSA', '허/XSA+ ᆫ/ETM', '혀/XSA', '답/XSA']
# xsv
# ['허/XSV', '만하/XSV', '하/XSV', '시키/XSV', '\uf537/XSV', '되/XSV', '히/XSV', '당허/XSV', '헤/XSV',
# '당하/XSV', '받/XSV', '거리/XSV', '혀/XSV', '__01/XSV']

idx_xpn = [115978, 1456, 139952, 139953, 328341, 403412, 192275, 365891, 365892, 119420, 26612, 26613, 74730, 383500, 139952, 139953, 35875, 68975, 161273, 161274, 252902, 166743, 166744, 166745, 134188, 401761, 88213, 88214, 397720, 312220, 312221, 368808, 314186, 151256, 170078, 81854, 324261, 224435, 337282, 42114, 407377, 407378, 127683, 268318, 268319, 203935, 203936, 12484, 415541, 340467, 374976]
idx_xsn = [1298, 1299, 1343, 1344, 559, 261, 1317, 237, 727, 1252, 238, 239, 240, 241, 242, 243, 244, 1253, 685, 883, 453, 454, 455, 456, 691, 692, 745, 746, 1244, 1333, 1325, 837, 786, 787, 217, 218, 233, 234, 235, 450, 800, 801, 262, 453, 454, 455, 456, 233, 234, 235, 259, 942, 943, 143, 144, 145, 146, 147, 148, 464, 689, 690, 1321, 1321, 1318, 1319, 1291, 1292, 1341, 737, 782, 783, 784, 230, 231, 1346, 1347, 1348, 188, 189, 190, 191, 192, 1245, 1246, 1192, 1193, 1195, 557, 558, 1263, 1264, 1265, 1266, 793, 889, 794, 795, 408, 257, 732, 933, 779, 694, 696, 823, 1296, 1302, 751, 1346, 1347, 1348, 186, 187, 1291, 219, 220, 1200, 1201, 1207]
idx_xsa = [813, 533, 686, 1336, 446]
idx_xsv = [1336, 828, 533, 451, 728, 168]

idx_xpn = []
idx_xsn = []
idx_xsa = []
idx_xsv = []
    # POS 태그 추가할 인덱스 얻기
def get_POS_idx(list, str):
    print(SDK_all_sorted[SDK_all_sorted["어휘_no_homonym"]==str])
    if len(SDK_all_sorted[SDK_all_sorted["어휘_no_homonym"] == str + '-']) >= 1:  # -기- 같은 어중 접미사 거르기용
        print("-x- Warning!!!")
        return None

    if len(SDK_all_sorted[SDK_all_sorted["어휘_no_homonym"]==str]) == 1:
        list.append(SDK_all_sorted[SDK_all_sorted["어휘_no_homonym"] == str].index.values.astype(int)[0])
    else:
        print("Warning!!!")
        return None

get_POS_idx(idx_xsn, '-추')
# idx_xsn.append(1351)

    # 사전에 POS 추가하기
def add_POS(df, idx_list, POS):
    real_idx_list = sorted(set(idx_list))   # 중복된 인덱스 제거
    for ix in range(len(real_idx_list)):
        df.iloc[real_idx_list[ix], -1].append(POS)

add_POS(SDK_all_sorted, idx_xpn, 'XPN')
add_POS(SDK_all_sorted, idx_xsn, 'XSN')
add_POS(SDK_all_sorted, idx_xsa, 'XSA')
add_POS(SDK_all_sorted, idx_xsv, 'XSV')


### 수동 교정 ###
    # 0) 연쇄(01)의 원어 정보: '連<equ>&#x9396;</equ>'
SDK_all_sorted[(SDK_all_sorted["어휘"] == "연쇄(01)")]
SDK_all_sorted.loc[255269, "원어"] = "連鎖"
    # 1) 연쇄점의 원어 정보: '連<equ>&#x9396;</equ>店'
SDK_all_sorted[(SDK_all_sorted["어휘"] == "연쇄-점")]
SDK_all_sorted.loc[255279, "원어"] = "連鎖店"
    # 2) 폐쇄의 원어 정보: '閉<equ>&#x9396;</equ>'
SDK_all_sorted[(SDK_all_sorted["어휘"] == "폐쇄")]
SDK_all_sorted.loc[398037, "원어"] = "閉鎖"
    # 3) <equ>&#x9396;</equ> -> 鎖
sai_ix = [ix for ix in range(len(SDK_all_sorted)) if (type(SDK_all_sorted.loc[ix, "원어"]) == str) and ('<equ>&#x9396;</equ>' in SDK_all_sorted.loc[ix, "원어"])]   # <equ>&#x9396;</equ> 있는 모든 행 확인
new_orig = [SDK_all_sorted.loc[ix,'원어'].replace('<equ>&#x9396;</equ>', '鎖') if (type(SDK_all_sorted.loc[ix, "원어"]) == str) and ('<equ>&#x9396;</equ>' in SDK_all_sorted.loc[ix, "원어"]) else SDK_all_sorted.loc[ix,'원어'] for ix in range(len(SDK_all_sorted))]
SDK_all_sorted['원어'] = new_orig
    # 4) 동티모르 -> 동-티모르
prob_ix = [ix for ix in range(len(SDK_all_sorted)) if (type(SDK_all_sorted.loc[ix, "어휘"]) == str) and ('동티모르' in SDK_all_sorted.loc[ix, "어휘"])]
SDK_all_sorted.iloc[101511, 0] = '동-티모르'
SDK_all_sorted.iloc[101511, 3] = '동-티모르'
    # 5) 핑곗거리: 원어 정보를 고유어로
prob_ix = [ix for ix in range(len(SDK_all_sorted)) if (type(SDK_all_sorted.loc[ix, "어휘"]) == str) and ('핑곗-거리' in SDK_all_sorted.loc[ix, "어휘"])]
SDK_all_sorted.iloc[405041, 7] = '고유어'
    # 6) 이솝우화: 이솝^우화
prob_ix = [ix for ix in range(len(SDK_all_sorted)) if (type(SDK_all_sorted.loc[ix, "어휘"]) == str) and ('이솝 우화' in SDK_all_sorted.loc[ix, "어휘"])]
SDK_all_sorted.iloc[prob_ix, 3] = '이솝^우화'
    # 7) 피마자의 ∇ > ▽
prob_ix = [ix for ix in range(len(SDK_all_sorted)) if (type(SDK_all_sorted.loc[ix, "어휘"]) == str) and (SDK_all_sorted.loc[ix, "어휘"] == "피마자")]
SDK_all_sorted.loc[prob_ix, "원어"] = '蓖▽麻子'

# save as a pickle file
# SDK_all_sorted.to_pickle("./output/parsed_dic_linux.pkl") # parsed, sorted dictionary
SDK_all_sorted.to_pickle("./output/parsed_dic_SD_linux_fixed.pkl") # parsed, sorted dictionary
