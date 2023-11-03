# make vocab, and config files for BERT pretraining
import argparse
import copy
import json
import os
import re


# get sub-paths for making BERT vocab, and config files
def get_paths(root_path: str):
    # all sub-paths
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(root_path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    # all paths of "tok.vocab"
    tok_vocab_paths = [path for path in listOfFiles if path.endswith("tok.vocab")]

    return tok_vocab_paths



# load bert_config.json
def load_bert_config(config_path:str, **kwargs):
    config_original_path = config_path
    with open(config_original_path) as json_file:
        bert_config = json.load(json_file)

    print(kwargs)

    if "model_max_length" in kwargs.keys():
        bert_config["model_max_length"] = kwargs["model_max_length"]

    return bert_config



# def main(root_path: str, vocab_size: int, model_max_length: int):
def main(root_path: str, model_max_length: int):
    tok_vocab_paths = get_paths(root_path=root_path)
    bert_config = load_bert_config(config_path="./resources/bert_config.json")   # for BERT training
    special_tokens_map = load_bert_config(config_path="./resources/special_tokens_map.json")    # for Hugging Face transformers
    tokenizer_config = load_bert_config(config_path="./resources/tokenizer_config.json", model_max_length=model_max_length) # for Hugging Face transformers

    regex_vocab_size = re.compile("(\d+)/$")

    for ix in range(len(tok_vocab_paths)):

        # output path
        output_dir = tok_vocab_paths[ix].split("tok.vocab")[0]

        # set vocab_size
        vocab_size = regex_vocab_size.search(output_dir).group(1)
        new_bert_config = copy.deepcopy(bert_config)
        new_bert_config["vocab_size"] = vocab_size

        # make "bert_config.json"
        with open(os.path.join(output_dir, "config.json"), "w") as json_file:  # KLUE 양식에 맞추기
            json.dump(new_bert_config, json_file, indent=4)

        with open(os.path.join(output_dir, "special_tokens_map.json"), "w") as json_file:  # KLUE 양식에 맞추기
            json.dump(special_tokens_map, json_file, indent=4)
        #
        with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as json_file:  # KLUE 양식에 맞추기
            json.dump(tokenizer_config, json_file, indent=4)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--model_max_length", type=int, default=128)

    args = vars(parser.parse_args())
    print(args)

    main(root_path=args["root_path"], model_max_length=args["model_max_length"])
