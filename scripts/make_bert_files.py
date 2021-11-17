# make vocab, and config files for BERT pretraining
import argparse
import json
import os


# get sub-paths for making BERT vocab, and config files
def get_paths(root_path: str):
    # all sub-paths
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(root_path):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    # all paths of "tok.vocab"
    tok_vocab_paths = [path for path in listOfFiles if path.endswith("tok.vocab")]

    return tok_vocab_paths


# # make a new bert_config
# def make_bert_config(vocab_size: int):
#     bert_config_original_path = "./resources/bert_config.json"
#     with open(bert_config_original_path) as json_file:
#         bert_config = json.load(json_file)
#
#     bert_config["vocab_size"] = vocab_size
#
#     return bert_config


# load bert_config.json
def load_bert_config(config_path:str, **kwargs):
    config_original_path = config_path
    with open(config_original_path) as json_file:
        bert_config = json.load(json_file)


    print(kwargs)

    if "vocab_size" in kwargs.keys():
        bert_config["vocab_size"] = kwargs["vocab_size"]

    if "model_max_length" in kwargs.keys():
        bert_config["model_max_length"] = kwargs["model_max_length"]


    return bert_config


# # load special_tokens_map.json
# def load_special_tokens_map():
#     config_original_path = "./resources/special_tokens_map.json"
#     with open(config_original_path) as json_file:
#         special_tokens_map = json.load(json_file)
#
#
#     return special_tokens_map
#
#
# # load bert_config.json
# def load_tokenizer_config(vocab_size: int):
#     config_original_path = "./resources/tokenizer_config.json"
#     with open(config_original_path) as json_file:
#         tokenizer_config = json.load(json_file)
#
#     tokenizer_config["vocab_size"] = vocab_size
#
#     return tokenizer_config






def main(root_path: str, vocab_size: int, model_max_length: int):
    tok_vocab_paths = get_paths(root_path=root_path)
    bert_config = load_bert_config(config_path="./resources/bert_config.json", vocab_size=vocab_size)
    special_tokens_map = load_bert_config(config_path="./resources/special_tokens_map.json")
    tokenizer_config = load_bert_config(config_path="./resources/tokenizer_config.json", model_max_length=model_max_length)


    for ix in range(len(tok_vocab_paths)):

        # delete number column
        output_dir = tok_vocab_paths[ix].split("tok.vocab")[0]

        # # make a "vocab.txt"
        # replaced = []
        #
        # with open(os.path.join(output_dir, "tok.vocab"), "r", encoding='utf-8') as f:
        #     text = f.readlines()
        #     for line in text:
        #         replaced.append(line.split('\t')[0])
        #
        # with open(os.path.join(output_dir, "vocab.txt"), "w", encoding='utf-8') as f:
        #     for vocab in replaced:
        #         f.write(vocab + '\n')


        # make a "bert_config.json"
        # with open(os.path.join(output_dir, "bert_config.json"), "w") as json_file:
        with open(os.path.join(output_dir, "config.json"), "w") as json_file:  # KLUE 양식에 맞추기
            json.dump(bert_config, json_file, indent=4)

        with open(os.path.join(output_dir, "special_tokens_map.json"), "w") as json_file:  # KLUE 양식에 맞추기
            json.dump(special_tokens_map, json_file, indent=4)
        #
        with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as json_file:  # KLUE 양식에 맞추기
            json.dump(tokenizer_config, json_file, indent=4)


    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=64000)
    parser.add_argument("--model_max_length", type=int, default=128)

    args = vars(parser.parse_args())
    print(args)

    main(root_path=args["root_path"], vocab_size=args["vocab_size"], model_max_length=args["model_max_length"])


    # root_path = "./resources/with_dummy_letter_v1"
    # vocab_size = 64000
    #
    # main(root_path=root_path, vocab_size=vocab_size)
