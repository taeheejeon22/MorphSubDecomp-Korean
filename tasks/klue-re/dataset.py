# coding=utf-8

import json
import logging
import os
from typing import List, Tuple

import torch
import transformers
from torch.utils.data import DataLoader, Dataset, TensorDataset
from utils import InputExample, InputFeatures

logger = logging.getLogger(__name__)


class KlueReProcessor:
    def __init__(self, args, tokenizer) -> None:

        self.hparams = args
        self.tokenizer = tokenizer

        # special tokens to mark the subject/object entity boundaries
        self.subject_start_marker = "<subj>"
        self.subject_end_marker = "</subj>"
        self.object_start_marker = "<obj>"
        self.object_end_marker = "</obj>"

        self.tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    self.subject_start_marker,
                    self.subject_end_marker,
                    self.object_start_marker,
                    self.object_end_marker,
                ]
            }
        )

        # Load relation class
        relation_class_file_path = os.path.join(
            self.hparams.data_dir, self.hparams.relation_filename
        )

        with open(relation_class_file_path, "r", encoding="utf-8") as f:
            self.relation_class = json.load(f)["relations"]

    def get_test_dataset(self, data_dir: str, file_name: str = None) -> Dataset:
        file_path = os.path.join(data_dir, file_name)

        assert os.path.exists(
            file_path
        ), "KlueReProcessor tries to open test file, but test dataset doesn't exists."

        logger.info(f"Loading from {file_path}")
        return self._create_dataset(file_path)

    def get_labels(self):
        return self.relation_class

    def _create_examples(self, file_path: str) -> List[InputExample]:
        examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            data_lst = json.load(f)

        for data in data_lst:
            guid = data["guid"]
            text = data["sentence"]
            subject_entity = data["subject_entity"]
            object_entity = data["object_entity"]
            label = data["label"]

            text = self._mark_entity_spans(
                text=text,
                subject_range=(
                    int(subject_entity["start_idx"]),
                    int(subject_entity["end_idx"]),
                ),
                object_range=(
                    int(object_entity["start_idx"]),
                    int(object_entity["end_idx"]),
                ),
            )
            examples.append(InputExample(guid=guid, text_a=text, label=label))

        return examples

    def _mark_entity_spans(
        self,
        text: str,
        subject_range: Tuple[int, int],
        object_range: Tuple[int, int],
    ) -> str:
        """
        Add entity markers to the text to identify the subject/object entities.

        Args:
            text: Original sentence
            subject_range: Pair of start and end indices of subject entity
            object_range: Pair of start and end indices of object entity

        Returns:
            A string of text with subject/object entity markers
        """
        if subject_range < object_range:
            segments = [
                text[: subject_range[0]],
                self.subject_start_marker,
                text[subject_range[0] : subject_range[1] + 1],
                self.subject_end_marker,
                text[subject_range[1] + 1 : object_range[0]],
                self.object_start_marker,
                text[object_range[0] : object_range[1] + 1],
                self.object_end_marker,
                text[object_range[1] + 1 :],
            ]
        elif subject_range > object_range:
            segments = [
                text[: object_range[0]],
                self.object_start_marker,
                text[object_range[0] : object_range[1] + 1],
                self.object_end_marker,
                text[object_range[1] + 1 : subject_range[0]],
                self.subject_start_marker,
                text[subject_range[0] : subject_range[1] + 1],
                self.subject_end_marker,
                text[subject_range[1] + 1 :],
            ]
        else:
            raise ValueError("Entity boundaries overlap.")

        marked_text = "".join(segments)

        return marked_text

    # copied from klue_baseline.data.utils.convert_examples_to_features
    def _convert_features(self, examples: List[InputExample]) -> List[InputFeatures]:
        max_length = self.hparams.max_seq_length
        if max_length is None:
            max_length = self.tokenizer.max_len

        label_map = {label: i for i, label in enumerate(self.get_labels())}
        labels = [label_map[example.label] for example in examples]

        def check_tokenizer_type():
            """
            Check tokenizer type.
            In KLUE paper, we only support wordpiece (BERT, KLUE-RoBERTa, ELECTRA) & sentencepiece (XLM-R).
            Will give warning if you use other tokenization. (e.g. bbpe)
            """
            if isinstance(self.tokenizer, transformers.XLMRobertaTokenizer):
                logger.info(
                    f"Using {type(self.tokenizer).__name__} for fixing tokenization result"
                )
                return "xlm-sp"  # Sentencepiece
            elif isinstance(self.tokenizer, transformers.BertTokenizer) or isinstance(
                self.tokenizer, transformers.BertTokenizerFast
            ):
                logger.info(
                    f"Using {type(self.tokenizer).__name__} for fixing tokenization result"
                )
                return (
                    "bert-wp"  # Wordpiece (including BertTokenizer & ElectraTokenizer)
                )
            else:
                logger.warn(
                    f"your tokenizer : {type(self.tokenizer).__name__}, If you are using other tokenizer (e.g. bbpe), you should change code in `fix_tokenization_error()`"
                )
                return "other"

        def fix_tokenization_error(text, tokenizer_type):
            tokens = self.tokenizer.tokenize(text)
            # subject
            if (
                text[text.find(self.subject_end_marker) + len(self.subject_end_marker)]
                != " "
            ):
                space_idx = tokens.index(self.subject_end_marker) + 1
                if tokenizer_type == "xlm-sp":
                    if tokens[space_idx] == "▁":
                        tokens.pop(space_idx)
                    elif tokens[space_idx].startswith("▁"):
                        tokens[space_idx] = tokens[space_idx][1:]
                elif tokenizer_type == "bert-wp":
                    if (
                        not tokens[space_idx].startswith("##")
                        and "가" <= tokens[space_idx][0] <= "힣"
                    ):
                        tokens[space_idx] = "##" + tokens[space_idx]

            # object
            if (
                text[text.find(self.object_end_marker) + len(self.object_end_marker)]
                != " "
            ):
                space_idx = tokens.index(self.object_end_marker) + 1
                if tokenizer_type == "xlm-sp":
                    if tokens[space_idx] == "▁":
                        tokens.pop(space_idx)
                    elif tokens[space_idx].startswith("▁"):
                        tokens[space_idx] = tokens[space_idx][1:]
                elif tokenizer_type == "bert-wp":
                    if (
                        not tokens[space_idx].startswith("##")
                        and "가" <= tokens[space_idx][0] <= "힣"
                    ):
                        tokens[space_idx] = "##" + tokens[space_idx]

            return tokens

        tokenizer_type = check_tokenizer_type()
        tokenized_examples = [
            fix_tokenization_error(example.text_a, tokenizer_type)
            for example in examples
        ]
        batch_encoding = self.tokenizer.batch_encode_plus(
            [
                (self.tokenizer.convert_tokens_to_ids(tokens), None)
                for tokens in tokenized_examples
            ],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        features = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}

            feature = InputFeatures(**inputs, label=labels[i])
            features.append(feature)

        for i in range(5):
            logger.info("*** Example ***")
            logger.info("guid: %s" % (examples[i].guid))
            logger.info("origin example: %s" % examples[i])
            logger.info(
                "origin tokens: %s" % self.tokenizer.tokenize(examples[i].text_a)
            )
            logger.info("fixed tokens: %s" % tokenized_examples[i])
            logger.info("features: %s" % features[i])

        return features

    def _create_dataset(self, file_path: str) -> Dataset:
        examples = self._create_examples(file_path)
        features = self._convert_features(examples)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor(
            [f.attention_mask for f in features], dtype=torch.long
        )
        # Some model does not make use of token type ids (e.g. RoBERTa)
        all_token_type_ids = torch.tensor(
            [0 if f.token_type_ids is None else f.token_type_ids for f in features],
            dtype=torch.long,
        )
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

        return TensorDataset(
            all_input_ids, all_attention_mask, all_token_type_ids, all_labels
        )


class KlueReDataLoader:
    def __init__(self, args, tokenizer):
        self.hparams = args
        self.processor = KlueReProcessor(args, tokenizer)

    def get_dataloader(
        self, batch_size: int, shuffle: bool = False, num_workers: int = 0
    ):
        return DataLoader(
            self.processor.get_test_dataset(
                self.hparams.data_dir, self.hparams.test_filename
            ),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )
