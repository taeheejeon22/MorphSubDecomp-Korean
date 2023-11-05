import argparse
import dataclasses
import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import torch
from overrides import overrides
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizer

from klue_baseline.data.base import DataProcessor, InputFeatures, KlueDataModule
from klue_baseline.data.utils import convert_examples_to_features

### our ###
import json
import inspect
import os
import sys

# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname( os.path.dirname(currentdir) )
# sys.path.insert(0, parentdir)

from tokenizer import (
    # CharTokenizer,
    # JamoTokenizer,
    # MeCabSentencePieceTokenizer_orig,
    # MeCabSentencePieceTokenizer_fixed,
    # MeCabSentencePieceTokenizer,
    MeCabWordPieceTokenizer,
    # MeCabTokenizer,
    # MeCabTokenizer_orig,
    # MeCabTokenizer_fixed,    # MeCabSentencePieceTokenizer_kortok,
    MeCabTokenizer_all,
    # MeCabTokenizer_kortok,
    # SentencePieceTokenizer,
    WordPieceTokenizer,
    Vocab,
    # WordTokenizer,
)
###


logger = logging.getLogger(__name__)


@dataclass
class KlueSTSInputExample:
    """
    A single example for KLUE-STS.

    Args:
        guid: Unique id for the example.
        text_a: The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
        text_b: The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
        score: The label of the example.
        binary_label: 0: False, 1: True
    """

    guid: str
    text_a: str
    text_b: str
    label: float
    binary_label: int

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json_string(self) -> str:
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2) + "\n"


class KlueSTSProcessor(DataProcessor):

    origin_train_file_name = "klue-sts-v1.1_train.json"
    origin_dev_file_name = "klue-sts-v1.1_dev.json"
    origin_test_file_name = "klue-sts-v1.1_test.json"

    datamodule_type = KlueDataModule

    # def __init__(self, args: argparse.Namespace, tokenizer: PreTrainedTokenizer) -> None:
    #     super().__init__(args, tokenizer)

    ### our ### morpheme pretokenizer
    def __init__(self, args: argparse.Namespace, tokenizer: PreTrainedTokenizer, path_rsc: str) -> None:
        super().__init__(args, tokenizer, path_rsc)

        with open(os.path.join(self.path_rsc, "tok.json")) as f:
            tokenizer_config: dict = json.load(f)

        self.pretokenizer = MeCabTokenizer_all(token_type=tokenizer_config["token_type"], tokenizer_type=tokenizer_config["tokenizer_type"], decomposition_type=tokenizer_config["decomposition_type"], space_symbol=tokenizer_config["space_symbol"], dummy_letter=tokenizer_config["dummy_letter"], nfd=tokenizer_config["nfd"], grammatical_symbol=tokenizer_config["grammatical_symbol"])
    ###

    @overrides
    def get_train_dataset(self, data_dir: str, file_name: Optional[str] = None) -> TensorDataset:
        file_path = os.path.join(data_dir, file_name or self.origin_train_file_name)

        logger.info(f"Loading from {file_path}")
        return self._create_dataset(file_path, "train")

    @overrides
    def get_dev_dataset(self, data_dir: str, file_name: Optional[str] = None) -> TensorDataset:
        file_path = os.path.join(data_dir, file_name or self.origin_dev_file_name)

        logger.info(f"Loading from {file_path}")
        return self._create_dataset(file_path, "dev")

    @overrides
    def get_test_dataset(self, data_dir: str, file_name: Optional[str] = None) -> TensorDataset:
        file_path = os.path.join(data_dir, file_name or self.origin_test_file_name)

        if not os.path.exists(file_path):
            logger.info("Test dataset doesn't exists. So loading dev dataset instead.")
            file_path = os.path.join(data_dir, self.hparams.dev_file_name or self.origin_dev_file_name)

        logger.info(f"Loading from {file_path}")
        return self._create_dataset(file_path, "test")

    @overrides
    def get_labels(self) -> List[str]:
        return []

    def _create_dataset(self, file_path: str, dataset_type: str) -> TensorDataset:
        examples = self._create_examples(file_path, dataset_type)
        features = self._convert_features(examples)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        # Some model does not make use of token type ids (e.g. RoBERTa)
        all_token_type_ids = torch.tensor(
            [0 if f.token_type_ids is None else f.token_type_ids for f in features], dtype=torch.long
        )
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        return TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)

    def _create_examples(self, file_path: str, dataset_type: str) -> List[KlueSTSInputExample]:
        examples = []
        with open(file_path, "r", encoding="utf=8") as f:
            data_lst = json.load(f)

        for data in data_lst:
            ## our
            text_a=data["sentence1"]
            text_b=data["sentence2"]
            ##

            examples.append(
                KlueSTSInputExample(
                    guid=data["guid"],
                    # text_a=data["sentence1"],
                    text_a = " ".join(self.pretokenizer.tokenize(text_a.strip())),  ### our ### pretokenization,
                    # text_b=data["sentence2"],
                    text_b = " ".join(self.pretokenizer.tokenize(text_b.strip())),  ### our ### pretokenization,
                    label=data["labels"]["real-label"],
                    binary_label=data["labels"]["binary-label"],
                )
            )


        ### our
        ### from klue_re.py ###
        for i in range(5):
            logger.info("*** Example ***")
            logger.info("guid: %s" % (examples[i].guid))
            # logger.info("origin example: %s" % examples[i].text_a)
            # logger.info("origin tokens: %s" % self.tokenizer.tokenize(examples[i].text_a))
            logger.info("origin example: %s" % data_lst[i]["sentence1"])
            logger.info("pretokenized example: %s" % self.pretokenizer.tokenize(data_lst[i]["sentence1"]))
            # logger.info("fixed tokens: %s" % tokenized_examples[i])
            logger.info("features: %s" % examples[i])
        ###


        return examples

    def _convert_features(self, examples: List[KlueSTSInputExample]) -> List[InputFeatures]:
        return convert_examples_to_features(
            examples,
            self.tokenizer,
            label_list=self.get_labels(),
            max_length=self.hparams.max_seq_length,
            task_mode="regression",
        )

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser, root_dir: str) -> argparse.ArgumentParser:
        parser = KlueDataModule.add_specific_args(parser, root_dir)
        parser.add_argument(
            "--max_seq_length",
            default=512,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )
        return parser
