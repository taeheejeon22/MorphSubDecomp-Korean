#!/bin/bash

set +x

OUTPUT_DIR="../run_outputs"
DATA_DIR="data/klue_benchmark"  # default submodule for data from https://github.com/KLUE-benchmark/KLUE
VERSION="v1.1"


tasks="ynat"

tokenizers=("eojeol_mecab_fixed_composed_grammatical_symbol_F_wp-64k" 
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

for tokenizer in "${tokenizers[@]}"; do
    # resource directory
    resource="../resources/v6_without_dummy_letter_grammatical_symbol_F/${tokenizer}"
    