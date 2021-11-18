#!/bin/bash

set +x

OUTPUT_DIR="klue_output"
DATA_DIR="data/klue_benchmark"  # default submodule for data from https://github.com/KLUE-benchmark/KLUE
VERSION="v1.1"

tasks="ynat"

resources=("eojeol_mecab_fixed_composed_grammatical_symbol_F_wp-64k" 
"eojeol_mecab_fixed_decomposed_pure_grammatical_symbol_F_wp-64k"
"morpheme_mecab_orig_composed__grammatical_symbol_F_wp-64k"
"morpheme_mecab_orig_decomposed_pure_grammatical_symbol_F_wp-64k"
"morpheme_mecab_fixed_composed_grammatical_symbol_F_wp-64k"
"morpheme_mecab_fixed_decomposed_pure_grammatical_symbol_F_wp-64k"
"morpheme_mecab_fixed_composed_grammatical_symbol_T_wp-64k"
"morpheme_mecab_fixed_decomposed_pure_grammatical_symbol_T_wp-64k"
"morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_T_wp-64k"
"morpheme_mecab_fixed_decomposed_grammatical_grammatical_symbol_T_wp-64k")

# 각 resource(tokenizer) 마다 task 수행

for model in "${resources[@]}"; do
    