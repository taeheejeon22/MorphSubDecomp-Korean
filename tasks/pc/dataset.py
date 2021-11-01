from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from tasks.bert_utils import convert_single_to_feature, pad_sequences
from tokenizer import BaseTokenizer, Vocab


class PCDataset(Dataset):
    """
    Dataset은 아래와 같은 Input 튜플을 가지고 있습니다.
    Index 0: input token ids
    Index 1: attention mask
    Index 2: token type ids
    Index 3: labels
    """

    def __init__(
        self, sentences: List[str], labels: List[int], vocab: Vocab, tokenizer: BaseTokenizer, max_sequence_length: int
    ):
        self.sentences = sentences
        self.labels = torch.tensor(labels)

        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

        self.bert_inputs = self._prepare_data(sentences)

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, item) -> Tuple[torch.Tensor, ...]:
        batch = (
            self.bert_inputs[0][item],
            self.bert_inputs[1][item],
            self.bert_inputs[2][item],
            self.labels[item],
        )
        return batch

    def _prepare_data(self, sentences: List[str]) -> Tuple[torch.Tensor, ...]:
        input_features = [
            convert_single_to_feature(sentence, self.tokenizer, self.vocab, self.max_sequence_length)
            for sentence in sentences
        ]

        padded_token_ids = torch.tensor(
            pad_sequences(
                [feature[0] for feature in input_features],
                padding_value=self.vocab.pad_token_id,
                max_length=self.max_sequence_length,
            ),
            dtype=torch.long,
        )
        padded_attention_mask = torch.tensor(
            pad_sequences(
                [feature[1] for feature in input_features], padding_value=0, max_length=self.max_sequence_length
            ),
            dtype=torch.long,
        )
        padded_token_type_ids = torch.tensor(
            pad_sequences(
                [feature[2] for feature in input_features], padding_value=0, max_length=self.max_sequence_length
            ),
            dtype=torch.long,
        )

        return (
            padded_token_ids,
            padded_attention_mask,
            padded_token_type_ids,
        )
