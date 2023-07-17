# https://github.com/irishev/BTS/blob/main/decompose_letters.py

# -*- coding: utf-8 -*-
import sys
import codecs
import hgtk
from tqdm import tqdm
from scripts.BTS.subchar_rule import subchar_dict, subchar_reverse_dict
import jamotools

END_CHAR = "/"
END_JAMO = "^"

def jamo_split(sentence, split_stroke=True, split_cji=True):
    result = []
    for word in sentence.split(' '):
        decomposed_word = ""
        for char in word:
            try:
                cho_joong_jong = hgtk.letter.decompose(char)
                char_seq = ""

                cho, joong, jong = cho_joong_jong
                cho = subchar_dict[cho] if split_stroke else cho
                joong = subchar_dict[joong] if split_cji else joong 
                jong = subchar_dict[jong] if split_stroke and len(jong) > 0 else jong
                cho += END_JAMO
                joong += END_JAMO
                jong += END_JAMO

                char_seq = cho + joong + jong
                decomposed_word += char_seq + END_CHAR

            except hgtk.exception.NotHangulException:
                got_exception=True
                decomposed_word += char
                continue
        result.append(decomposed_word.strip())
    return " ".join(result)

def jamo_split_original(sentence, split_stroke=True, split_cji=True):
    result = []
    for word in sentence.split(' '):
        decomposed_word = ""
        for char in word:
            try:
                cho_joong_jong = hgtk.letter.decompose(char)
                char_seq = ""

                cho, joong, jong = cho_joong_jong
                jong = jong if len(jong) > 0 else "^"

                char_seq = cho + joong + jong
                decomposed_word += char_seq

            except hgtk.exception.NotHangulException:
                got_exception=True
                decomposed_word += char
                continue
        result.append(decomposed_word.strip())
    return " ".join(result)

def reconstruct_word(word):
    chars = word.split(END_CHAR)
    jamo_word = ""
    for char in chars:
        jamos = char.split(END_JAMO)
        for jamo in jamos:
            try:
                jamo = subchar_reverse_dict[jamo]
                jamo_word += jamo
            except KeyError:
                jamo_word += jamo
        jamo_word += "&"
    return hgtk.text.compose(jamo_word, compose_code="&")

def reconstruct_word_jamo(word):
    word = word.replace("^", "")
    return jamotools.join_jamos(word)
        

def char_split(sentence):
    result = []
    for word in sentence.split(' '):
        decomposed_word = ""
        for char in word:
            try:
                cho_joong_jong = hgtk.letter.decompose(char)
                decomposed_word += char + END_CHAR
            except hgtk.exception.NotHangulException:
                decomposed_word += char
                continue
        result.append(decomposed_word)
    return " ".join(result)


def main():
    INPUT_FILE_PATH = sys.argv[1]
    OUTPUT_FILE_PATH = sys.argv[2]
    SPLIT_STROKE = eval(sys.argv[3])
    SPLIT_CJI = eval(sys.argv[4])
    num_lines = 0

    with codecs.open(OUTPUT_FILE_PATH, 'w', encoding='utf8') as jamo:
        with codecs.open(INPUT_FILE_PATH, 'r', encoding="utf8") as input_:
            for sentence in tqdm(input_, desc="Parsing Text", ncols=100, total=19270697):
                jamo_sentences = jamo_split(sentence.strip(), split_stroke=SPLIT_STROKE, split_cji=SPLIT_CJI)
                jamo.write(jamo_sentences + '\n')
                num_lines+=1

if __name__ == '__main__':
    main()


    
