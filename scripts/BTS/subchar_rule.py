# https://github.com/irishev/BTS/blob/main/subchar_rule.py

ADD_LINE = '-'
END_CHAR = 'c'


subchar_dict = {
    # 초성/종성 공통
    'ㄱ':'ㄱ', 
    'ㄲ':'ㄱㄱ', 
    'ㄴ':'ㄴ', 
    'ㄷ':'ㄴ'+ADD_LINE, 
    'ㄸ':'ㄴ'+ADD_LINE+'ㄴ'+ADD_LINE, 
    'ㄹ':'ㄹ', 
    'ㅁ':'ㅁ', 
    'ㅂ':'ㅁ'+ADD_LINE, 
    'ㅃ':'ㅁ'+ADD_LINE+'ㅁ'+ADD_LINE, 
    'ㅅ':'ㅅ', 
    'ㅆ':'ㅅㅅ', 
    'ㅇ':'ㅇ', 
    'ㅈ':'ㅅ'+ADD_LINE, 
    'ㅉ':'ㅅ'+ADD_LINE+'ㅅ'+ADD_LINE, 
    'ㅊ':'ㅅ'+ADD_LINE+ADD_LINE, 
    'ㅋ':'ㄱ'+ADD_LINE, 
    'ㅌ':'ㄴ'+ADD_LINE+ADD_LINE, 
    'ㅍ':'ㅁ'+ADD_LINE+ADD_LINE, 
    'ㅎ':'ㅇ'+ADD_LINE,
    # 중성
    'ㅏ':'ㅣ'+'ㆍ', 
    'ㅐ':'ㅣ'+'ㆍ'+'ㅣ', 
    'ㅑ':'ㅣ'+'ㆍ'+'ㆍ', 
    'ㅒ':'ㅣ'+'ㆍ'+'ㆍ'+'ㅣ', 
    'ㅓ':'ㆍ'+'ㅣ', 
    'ㅔ':'ㆍ'+'ㅣ'+'ㅣ', 
    'ㅕ':'ㆍ'+'ㆍ'+'ㅣ', 
    'ㅖ':'ㆍ'+'ㆍ'+'ㅣ'+'ㅣ', 
    'ㅗ':'ㆍ'+'ㅡ', 
    'ㅘ':'ㆍ'+'ㅡ'+'ㅣ'+'ㆍ', 
    'ㅙ':'ㆍ'+'ㅡ'+'ㅣ'+'ㆍ'+'ㅣ', 
    'ㅚ':'ㆍ'+'ㅡ'+'ㅣ', 
    'ㅛ':'ㆍ'+'ㆍ'+'ㅡ', 
    'ㅜ':'ㅡ'+'ㆍ', 
    'ㅝ':'ㅡ'+'ㆍ'+'ㆍ'+'ㅣ', 
    'ㅞ':'ㅡ'+'ㆍ'+'ㆍ'+'ㅣ'+'ㅣ', 
    'ㅟ':'ㅡ'+'ㆍ'+'ㅣ', 
    'ㅠ':'ㅡ'+'ㆍ'+'ㆍ', 
    'ㅡ':'ㅡ', 
    'ㅢ':'ㅡ'+'ㅣ', 
    'ㅣ':'ㅣ',
    # 종성 단독
    '':'', 
    'ㄳ':'ㄱㅅ', 
    'ㄵ':'ㄴㅅ'+ADD_LINE, 
    'ㄶ':'ㄴㅇ'+ADD_LINE, 
    'ㄺ':'ㄹㄱ', 
    'ㄻ':'ㄹㅁ', 
    'ㄼ':'ㄹㅁ'+ADD_LINE, 
    'ㄽ':'ㄹㅅ', 
    'ㄾ':'ㄹㄴ'+ADD_LINE+ADD_LINE, 
    'ㄿ':'ㄹㅁ'+ADD_LINE+ADD_LINE, 
    'ㅀ':'ㄹㅇ'+ADD_LINE,
    'ㅄ':'ㅁ'+ADD_LINE+'ㅅ', 
}

subchar_reverse_dict = {val:key for key, val in subchar_dict.items()}
