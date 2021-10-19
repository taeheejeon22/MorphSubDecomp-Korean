# 수정한 부분
# from tokenizer import
# def main()의 if config.tokenzer.startswith("mecab-"): 부분

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig

from tasks.bert_utils import load_pretrained_bert
from tasks.logger import get_logger
from tasks.cola.config import TrainConfig
from tasks.cola.data_utils import load_data
from tasks.cola.dataset import COLADataset
from tasks.cola.model import COLAModel
from tasks.cola.trainer import Trainer
from tokenizer import (
    # CharTokenizer,
    # JamoTokenizer,
    MeCabSentencePieceTokenizer,
    MeCabTokenizer,
    MeCabSentencePieceTokenizer_kortok,
    MeCabTokenizer_kortok,
    SentencePieceTokenizer,
    Vocab,
    # WordTokenizer,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    # config
    config = TrainConfig(**args)
    config = config._replace(
        log_dir=config.log_dir.format(config.tokenizer),
        summary_dir=config.summary_dir.format(config.tokenizer),
        # checkpoint_dir=config.checkpoint_dir.format(config.tokenizer),
    )
    set_seed(config.seed)

    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.summary_dir, exist_ok=True)
    # os.makedirs(config.checkpoint_dir, exist_ok=True)

    # logger
    logger = get_logger(log_path=os.path.join(config.log_dir, "logs.txt"))
    logger.info(config)

    # 기본적인 모듈들 생성 (vocab, tokenizer)
    tokenizer_dir = os.path.join(config.resource_dir, config.tokenizer)
    logger.info(f"get vocab and tokenizer from {tokenizer_dir}")
    vocab = Vocab(os.path.join(tokenizer_dir, "tok.vocab"))
    # if config.tokenizer.startswith("mecab-"):
    #     tokenizer = MeCabTokenizer(os.path.join(tokenizer_dir, "tok.json"))
    if config.tokenizer.startswith("sp-"):
        tokenizer = SentencePieceTokenizer(os.path.join(tokenizer_dir, "tok.model"))
    elif config.tokenizer.startswith("mecab_"):
        if args["use_kortok"] == False:
            mecab = MeCabTokenizer(os.path.join(tokenizer_dir, "tok.json"))
            sp = SentencePieceTokenizer(os.path.join(tokenizer_dir, "tok.model"))
            tokenizer = MeCabSentencePieceTokenizer(mecab, sp)

        elif args["use_kortok"] == True:
            mecab = MeCabTokenizer_kortok(os.path.join(tokenizer_dir, "tok.json"))
            sp = SentencePieceTokenizer(os.path.join(tokenizer_dir, "tok.model"))
            tokenizer = MeCabSentencePieceTokenizer_kortok(mecab, sp)



    # elif config.tokenizer.startswith("char-"):
    #     tokenizer = CharTokenizer()
    # elif config.tokenizer.startswith("word-"):
    #     tokenizer = WordTokenizer()
    # elif config.tokenizer.startswith("jamo-"):
    #     tokenizer = JamoTokenizer()
    else:
        raise ValueError("Wrong tokenizer name.")

    # 모델에 넣을 데이터 준비
    # label-to-index
    label_to_index = {"0": 0, "1": 1}
    # Train
    logger.info(f"read train data from {config.train_path}")
    train_sentences, train_labels = load_data(config.train_path, label_to_index)
    # Dev
    logger.info(f"read dev data from {config.dev_path}")
    dev_sentences, dev_labels = load_data(config.dev_path, label_to_index)
    # Test
    logger.info(f"read test data from {config.test_path}")
    test_sentences, test_labels = load_data(config.test_path, label_to_index)

    # 데이터로 dataloader 만들기
    # Train
    logger.info("create data loader using train data")
    train_dataset = COLADataset(train_sentences, train_labels, vocab, tokenizer, config.max_sequence_length)
    train_random_sampler = RandomSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset, sampler=train_random_sampler, batch_size=config.batch_size)
    # Dev
    logger.info("create data loader using dev data")
    dev_dataset = COLADataset(dev_sentences, dev_labels, vocab, tokenizer, config.max_sequence_length)
    dev_data_loader = DataLoader(dev_dataset, batch_size=1024)
    # Test
    logger.info("create data loader using test data")
    test_dataset = COLADataset(test_sentences, test_labels, vocab, tokenizer, config.max_sequence_length)
    test_data_loader = DataLoader(test_dataset, batch_size=1024)

    # Summary Writer 준비
    summary_writer = SummaryWriter(log_dir=config.summary_dir)

    # 모델을 준비하는 코드
    logger.info("initialize model and convert bert pretrained weight")
    bert_config = BertConfig.from_json_file(
        os.path.join(config.resource_dir, config.tokenizer, config.bert_config_file_name)
    )
    model = COLAModel(bert_config, config.dropout_prob)
    model.bert = load_pretrained_bert(
        bert_config, os.path.join(config.resource_dir, config.tokenizer, config.pretrained_bert_file_name)
    )

    trainer = Trainer(config, model, train_data_loader, dev_data_loader, test_data_loader, logger, summary_writer)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--desc", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--tokenizer", type=str)

    parser.add_argument("--resource_dir", type=str)
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--dev_path", type=str)
    parser.add_argument("--test_path", type=str)

    parser.add_arguement("--use_kortok", nargs="?", const=False, type=bool, default=False)  # kortok 토크나이저 사용 여부

    args = {k: v for k, v in vars(parser.parse_args()).items() if v}

    main(args)
