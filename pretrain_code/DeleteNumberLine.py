# 토크나이징을 통해 만들어진 tok.vocab을 BERT pre-training에 사용할 수 있도록
# 숫자 칼럼을 지우고 토큰만 남긴 vocab.txt 파일을 생성하는 코드.
# 사용법: tok.vocab이 있는 디렉토리에서,
# (bash) python3 DeleteNumberLine.py --vocab_file tok.vocab

import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", type=str, required=True) # vocab dir
    args = vars(parser.parse_args())
    print(args)


    replaced = []
    # delete number column
    output_dir = args["vocab_file"]
    with open(output_dir, "r", encoding='utf-8') as f:
        text = f.readlines()
        for line in text:
            replaced.append(line.split('\t')[0])
            
    with open('vocab.txt', "w", encoding='utf-8') as f:
        for vocab in replaced:
            f.write(vocab+'\n')

