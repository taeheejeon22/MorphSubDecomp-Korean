# BERT pretrain용 vocab 파일 자동 생성
import os

os.listdir("./resources/with_dummy_letter")

os.walk("./resources/with_dummy_letter")


# all sub-paths
listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk("./resources/with_dummy_letter"):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]

# all paths of "tok.vocab"
tok_vocab_paths = [path for path in listOfFiles if path.endswith("tok.vocab")]


for ix in range(len(tok_vocab_paths)):
    replaced = []
    # delete number column
    output_dir = tok_vocab_paths[ix].split("tok.vocab")[0]
    with open(output_dir + "tok.vocab", "r", encoding='utf-8') as f:
        text = f.readlines()
        for line in text:
            replaced.append(line.split('\t')[0])

    with open(output_dir + 'vocab.txt', "w", encoding='utf-8') as f:
        for vocab in replaced:
            f.write(vocab + '\n')
