import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig

import json
import sys
sys.path.insert(0, '.')

from tasks.bert_utils import load_pretrained_bert
from tasks.korsts.config import TrainConfig
from tasks.korsts.data_utils import load_data
from tasks.korsts.dataset import KorSTSDataset
from tasks.korsts.model import KorSTSModel
from tasks.korsts.trainer_test import Trainer
from tasks.logger import get_logger
from tokenizer import (
    # CharTokenizer,
    # JamoTokenizer,
    MeCabSentencePieceTokenizer_orig,
    MeCabSentencePieceTokenizer_fixed,
    # MeCabTokenizer,
    MeCabTokenizer_orig,
    MeCabTokenizer_fixed,    # MeCabSentencePieceTokenizer_kortok,
    # MeCabTokenizer_kortok,
    SentencePieceTokenizer,
    Vocab,
    # WordTokenizer,
)

from time import gmtime, strftime

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
os.environ["TOKENIZERS_PARALLELISM"] = "true"


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
        #checkpoint_dir=config.checkpoint_dir.format(config.tokenizer),
    )
    set_seed(config.seed)

    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.summary_dir, exist_ok=True)
    #os.makedirs(config.checkpoint_dir, exist_ok=True)


    # bert 모델 경로 자동 지정
    tokenizer_dir = os.path.join(config.resource_dir, config.tokenizer)
    pretrained_bert_files = [file for file in os.listdir(tokenizer_dir) if file.endswith("pth")]

    assert (len(pretrained_bert_files) == 1), 'There are more than one bert model files!!!!!!!'


    # logger
    # logger = get_logger(log_path=os.path.join(config.log_dir, "logs.txt"))
    pretrained_bert_file_name = pretrained_bert_files[0]
    begin_time = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
    logger = get_logger(log_path=os.path.join(config.log_dir, f"logs_{pretrained_bert_file_name}_{begin_time}.txt"))

    logger.info(config)

    # 기본적인 모듈들 생성 (vocab, tokenizer)
    logger.info(f"get vocab and tokenizer from {tokenizer_dir}")
    vocab = Vocab(os.path.join(tokenizer_dir, "tok.vocab"))
    # if config.tokenizer.startswith("mecab-"):
    #     tokenizer = MeCabTokenizer(os.path.join(tokenizer_dir, "tok.json"))

    # resource 경로 확인용
    print("\ntokenizer:", config.tokenizer)
    print("resources path:", tokenizer_dir, "\n")


    if config.tokenizer.startswith("sp-"):
        tokenizer = SentencePieceTokenizer(os.path.join(tokenizer_dir, "tok.model"))
    elif config.tokenizer.startswith("mecab_"):
        with open(os.path.join(tokenizer_dir, "tok.json")) as f:
            tokenizer_config: dict = json.load(f)

        # mecab = MeCabTokenizer(os.path.join(tokenizer_dir, "tok.json"))
        # mecab = MeCabTokenizer_fixed(tokenizer_type=tokenizer_config["tokenizer_type"], decomposition_type=tokenizer_config["decomposition_type"], space_symbol=tokenizer_config["space_symbol"], dummy_letter=tokenizer_config["dummy_letter"])
        sp = SentencePieceTokenizer(os.path.join(tokenizer_dir, "tok.model"))

        if "orig" in config.tokenizer:
            mecab = MeCabTokenizer_orig(tokenizer_type=tokenizer_config["tokenizer_type"], decomposition_type=tokenizer_config["decomposition_type"], space_symbol=tokenizer_config["space_symbol"], dummy_letter=tokenizer_config["dummy_letter"])
            tokenizer = MeCabSentencePieceTokenizer_orig(mecab, sp, use_fixed=False)
        elif "fixed" in config.tokenizer:
            mecab = MeCabTokenizer_fixed(tokenizer_type=tokenizer_config["tokenizer_type"], decomposition_type=tokenizer_config["decomposition_type"], space_symbol=tokenizer_config["space_symbol"], dummy_letter=tokenizer_config["dummy_letter"])
            tokenizer = MeCabSentencePieceTokenizer_fixed(mecab, sp, use_fixed=True)

        # elif args["use_kortok"] == True:
        #     print("use_kortok: ", args["use_kortok"])
        #     mecab = MeCabTokenizer_kortok(os.path.join(tokenizer_dir, "tok.json"))
        #     sp = SentencePieceTokenizer(os.path.join(tokenizer_dir, "tok.model"))
        #     tokenizer = MeCabSentencePieceTokenizer_kortok(mecab, sp)

    # elif config.tokenizer.startswith("char-"):
    #     tokenizer = CharTokenizer()
    # elif config.tokenizer.startswith("word-"):
    #     tokenizer = WordTokenizer()
    # elif config.tokenizer.startswith("jamo-"):
    #     tokenizer = JamoTokenizer()
    else:
        raise ValueError("Wrong tokenizer name.")

    # 모델에 넣을 데이터 준비
    # Train
    logger.info(f"read training data from {config.train_path}")
    train_sentence_as, train_sentence_bs, train_labels = load_data(config.train_path)
    # Dev
    logger.info(f"read dev data from {config.dev_path}")
    dev_sentence_as, dev_sentence_bs, dev_labels = load_data(config.dev_path)
    # Test
    logger.info(f"read test data from {config.test_path}")
    test_sentence_as, test_sentence_bs, test_labels = load_data(config.test_path)


    # 토큰화 데모
    print(f"tokenization sample 1: {tokenizer.tokenize(train_sentence_as[0])}")
    print(f"tokenization sample 2: {tokenizer.tokenize(train_sentence_bs[0])}")


    # 데이터로 dataloader 만들기
    # Train
    logger.info("create data loader using training data")
    train_dataset = KorSTSDataset(
        train_sentence_as, train_sentence_bs, train_labels, vocab, tokenizer, config.max_sequence_length
    )
    train_random_sampler = RandomSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset, sampler=train_random_sampler, batch_size=config.batch_size)
    # Dev
    logger.info("create data loader using dev data")
    dev_dataset = KorSTSDataset(
        dev_sentence_as, dev_sentence_bs, dev_labels, vocab, tokenizer, config.max_sequence_length
    )
    dev_data_loader = DataLoader(dev_dataset, batch_size=1024)
    # Test
    logger.info("create data loader using test data")
    test_dataset = KorSTSDataset(
        test_sentence_as, test_sentence_bs, test_labels, vocab, tokenizer, config.max_sequence_length
    )
    test_data_loader = DataLoader(test_dataset, batch_size=1024)

    # Summary Writer 준비
    summary_writer = SummaryWriter(log_dir=config.summary_dir)

    # 모델을 준비하는 코드
    logger.info("initialize model and convert bert pretrained weight")
    bert_config = BertConfig.from_json_file(
        os.path.join(config.resource_dir, config.tokenizer, config.bert_config_file_name)
    )
    model = KorSTSModel(bert_config, config.dropout_prob)
    # model.bert = load_pretrained_bert(
    #     bert_config, os.path.join(config.resource_dir, config.tokenizer, config.pretrained_bert_file_name)
    # )
    model.bert = load_pretrained_bert(
        bert_config, os.path.join(config.resource_dir, config.tokenizer, pretrained_bert_file_name)
    )

    # device, model 정의
    if config.use_tpu == "tpu":
        # 사전에 torch_xla 설치 필요
        device = xm.xla_device()
        model = model
        print('TPU running...')
    elif config.use_tpu == "gpu":    
        # multi gpu(3)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
            print('Multi GPU({}) activate'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model, device_ids=[0,1,2,3])
        else:
            model = model


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


    # hyperparameters
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)

    # use tpu
    parser.add_argument("--use_tpu", type=str, default="gpu")

    #log, summary dir
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--summary_dir", type=str)

    args = {k: v for k, v in vars(parser.parse_args()).items() if v}



    xmp.spawn(main(args), nprocs=8)
    print('multi run...')
    # else:
    #     print('not tpu...')
    #     main(args)