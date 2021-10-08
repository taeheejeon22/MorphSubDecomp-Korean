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


# make a new bert config file
def make_bert_config(vocab_size: int):
    bert_config_original_path = "./resources/bert_config.json"
    with open(bert_config_original_path) as json_file:
        bert_config = json.load(json_file)

    bert_config["vocab_size"] = vocab_size

    return bert_config


def main(root_path: str, vocab_size: int):
    tok_vocab_paths = get_paths(root_path=root_path)
    bert_config = make_bert_config(vocab_size=vocab_size)

    for ix in range(len(tok_vocab_paths)):
        # make a "vocab.txt"
        replaced = []

        # delete number column
        output_dir = tok_vocab_paths[ix].split("tok.vocab")[0]

        # with open(output_dir + "tok.vocab", "r", encoding='utf-8') as f:
        with open(os.path.join(output_dir, "tok.vocab"), "r", encoding='utf-8') as f:
            text = f.readlines()
            for line in text:
                replaced.append(line.split('\t')[0])

        # with open(output_dir + "vocab.txt", "w", encoding='utf-8') as f:
        with open(os.path.join(output_dir, "vocab.txt"), "w", encoding='utf-8') as f:
            for vocab in replaced:
                f.write(vocab + '\n')

        # make a "bert_config.json"
        # with open(output_dir + "bert_config.json", "w") as json_file:
        with open(os.path.join(output_dir, "bert_config.json"), "w") as json_file:
            json.dump(bert_config, json_file)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=64000)

    args = vars(parser.parse_args())
    print(args)

    main(root_path=args["root_path"], vocab_size=args["vocab_size"])


    # root_path = "./resources/with_dummy_letter_v1"
    # vocab_size = 64000
    #
    # main(root_path=root_path, vocab_size=vocab_size)
