#!/bin/bash


tokenizers=("morpheme_mecab_fixed_decomposed_lexical_grammatical_symbol_F_wp-64k" 
"morpheme_mecab_fixed_decomposed_pure_grammatical_symbol_F_wp-64k"
"morpheme_mecab_fixed_decomposed_grammatical_grammatical_symbol_F_wp-64k" 
"morpheme_mecab_orig_composed_grammatical_symbol_F_wp-64k" 
"morpheme_mecab_orig_decomposed_pure_grammatical_symbol_F_wp-64k")

for tokenizer in "${tokenizers[@]}"; do
    echo "### tokenizer: ${tokenizer}"

done



tokenizers=("morpheme_mecab_fixed_decomposed_grammatical_grammatical_symbol_F_wp-64k" 
"morpheme_mecab_orig_composed_grammatical_symbol_F_wp-64k" 
"morpheme_mecab_orig_decomposed_pure_grammatical_symbol_F_wp-64k")


for tokenizer in "${tokenizers[@]}"; do
    echo "### tokenizer: ${tokenizer}"

done