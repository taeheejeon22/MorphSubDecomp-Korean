"""Usage
$ python inference.py --data_dir data \
                      --model_dir model \
                      --output_dir output \
                      [args..]
"""
import argparse
import logging
import os
import tarfile

import torch
from dataset import KlueReDataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

KLUE_RE_OUTPUT = "output.csv"


def load_model(model_dir, model_tar_path):
    """load model from tar file pre-fetched from s3

    Args:
        model_dir: str: the directory of tar file
        model_tar_path: str: the name of tar file
    """
    tar = tarfile.open(model_tar_path, "r:gz")
    tar.extractall(path=model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    return model


@torch.no_grad()
def inference(args) -> None:

    data_dir = args.data_dir
    model_dir = args.model_dir
    model_tar_path = os.path.join(model_dir, args.model_tarname)
    output_dir = args.output_dir

    assert os.path.exists(
        data_dir
    ), "Run inference code w/o data folder. Plz check out the path of data"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = open(os.path.join(output_dir, KLUE_RE_OUTPUT), "w")

    # configure gpu
    num_gpus = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model via transformer.AutoModelForSequenceClassification")
    # load model
    model = load_model(model_dir, model_tar_path).to(device)
    if num_gpus > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    # load tokenizer
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

    logger.info("Data Loader : preprocessing data")
    data_loader = KlueReDataLoader(args, tokenizer).get_dataloader(
        args.batch_size, num_workers=args.num_workers
    )

    logger.info("Start inferencing")
    for batch in data_loader:
        input_ids, attention_mask, token_type_ids, labels = [
            b.to(device) for b in batch
        ]

        output = model(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        logits = output[0]

        preds, probs = (
            torch.argmax(logits, dim=1).detach().cpu().numpy(),
            torch.softmax(logits, dim=1).detach().cpu().numpy(),
        )

        for i in range(len(preds)):
            output_file.write(f"{preds[i]}\t{' '.join(map(str,probs[i].tolist()))}\n")

    output_file.close()
    logger.info("Done inferencing")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--data_dir", type=str, default=os.environ.get("SM_CHANNEL_EVAL", "/data")
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./model",
    )
    parser.add_argument(
        "--model_tarname",
        type=str,
        default="klue-re.tar.gz",
        help="it needs to include all things for loading baseline model & tokenizer, \
             only supporting transformers.AutoModelForSequenceClassification as a model \
             transformers.XLMRobertaTokenizer or transformers.BertTokenizer as a tokenizer",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/output"),
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=510,
        help="maximum sequence length (default: 510)",
    )
    parser.add_argument(
        "--relation_filename",
        default="relation_list.json",
        type=str,
        help="File name of list of relation classes (default: relation_list.json)",
    )
    parser.add_argument(
        "--test_filename",
        default="klue-re-v1.1_test.json",
        type=str,
        help="Name of the test file (default: klue-re-v1.1_test.json)",
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="kwarg passed to DataLoader"
    )

    args = parser.parse_args()

    inference(args)


if __name__ == "__main__":
    main()
