#!/usr/bin/python3

# https://gigglehd.com/zbxe/14052329

## 참조문헌 :
## http://d2.naver.com/helloworld/76650
## http://dream.ahboom.net/i/entry/28
## http://www.unicode.org/charts/PDF/U1100.pdf

chosung = (
	"ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ",
	"ㅃ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ",
	"ㅌ", "ㅍ", "ㅎ")

jungsung = (
	"ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ",
	"ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ",
	"ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ")

jongsung = (
	"", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ",
	"ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",
	"ㅁ", "ㅂ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ",
	"ㅋ", "ㅌ", "ㅍ", "ㅎ")

def isHangeul(one_character):
	"""글자 하나를 입력받아, 이것이 조합된 한글인지 판단한다."""
	return 0xAC00 <= ord(one_character[:1]) <= 0xD7A3

def hangeulExplode(one_hangeul):
	"""조합된 한글 글자 하나를 입력받아, 이를 초·중·종성으로 분해한다."""
	a = one_hangeul[:1]
	if isHangeul(a) != True:
		return False
	b = ord(a) - 0xAC00
	cho = b // (21*28)
	jung = b % (21*28) // 28
	jong = b % 28
	if jong == 0:
		return (chosung[cho], jungsung[jung])
	else:
		return (chosung[cho], jungsung[jung], jongsung[jong])

def hangeulJoin(inputlist):
	"""분해된 한글 낱자들을 리스트로 입력받아, 이를 조합하여 모아쓰기된 문장으로 바꾼다."""
	result = ""
	cho, jung, jong = 0, 0, 0
	inputlist.insert(0, "")
	while len(inputlist) > 1:
		if inputlist[-1] in jongsung:
			if inputlist[-2] in jungsung:
				jong = jongsung.index(inputlist.pop())
			else:
				result += inputlist.pop()
		elif inputlist[-1] in jungsung:
			if inputlist[-2] in chosung:
				jung = jungsung.index(inputlist.pop())
				cho = chosung.index(inputlist.pop())
				result += chr(0xAC00 + ((cho*21)+jung)*28+jong)
				cho, jung, jong = 0, 0, 0
			else:
				result += inputlist.pop()
		else:
			result += inputlist.pop()
	else:
		return result[::-1]

def pureosseugi(inputtext):
	"""입력된 문장에 있는 모든 한글 글자를 풀어쓰기로 바꾼다."""
	result = ""
	for i in inputtext:
		if isHangeul(i) == True:
			for j in hangeulExplode(i):
				result += j
		else:
			result += i
	return result

def moasseugi(inputtext):
	"""입력된 문장에 있는 모든 풀어쓰기된 한글을 모아쓰기로 바꾼다."""
	t1 = []
	for i in inputtext:
		t1.append(i)
	return hangeulJoin(t1)
