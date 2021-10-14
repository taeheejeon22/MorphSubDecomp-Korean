from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from tasks.bert_utils import convert_pair_to_feature, pad_sequences
from tokenizer import BaseTokenizer, Vocab


class PAWSDataset(Dataset):
    """
    Dataset은 아래와 같은 Input 튜플을 가지고 있습니다.
    Index 0: input token ids
    Index 1: attentio mask
    Index 2: token type ids
    Index 3: labels
    """

    def __init__(
        self,
        sentence_as: List[str],
        sentence_bs: List[str],
        labels: List[int],
        vocab: Vocab,
        tokenizer: BaseTokenizer,
        max_sequence_length: int,
    ):
        self.sentence_as = sentence_as
        self.sentence_bs = sentence_bs
        self.labels = torch.tensor(labels)

        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

        bert_inputs = _prepare_data(sentence_as, sentence_bs)


        sentence_as = train_sentence_as[:]
        sentence_bs = train_sentence_bs[:]



    def __len__() -> int:
        return labels.size(0)

    def __getitem__(item) -> Tuple[torch.Tensor, ...]:
        batch = (
            bert_inputs[0][item],
            bert_inputs[1][item],
            bert_inputs[2][item],
            labels[item],
        )
        return batch

    def _prepare_data(sentence_as: List[str], sentence_bs: List[str]) -> Tuple[torch.Tensor, ...]:
        input_features = [
            convert_pair_to_feature(sentence_a, sentence_b, tokenizer, vocab, max_sequence_length)
            for sentence_a, sentence_b in zip(sentence_as, sentence_bs)
        ]

        ee = list(zip(sentence_as, sentence_bs))
        # for ix in range(len(ee)):   # 823
        for ix in range(824, len(ee)):  # 823
            # convert_pair_to_feature(ee[ix][0], re.sub(p_kakao, "EE", ee[ix][1]), tokenizer, vocab, 128)
            convert_pair_to_feature(ee[ix][0], ee[ix][1], tokenizer, vocab, 128)

            sentence_a = ee[ix][0]
            sentence_b = ee[ix][1]






        padded_token_ids = torch.tensor(
            pad_sequences(
                [feature[0] for feature in input_features],
                padding_value=vocab.pad_token_id,
                max_length=max_sequence_length,
            ),
            dtype=torch.long,
        )
        padded_attention_mask = torch.tensor(
            pad_sequences(
                [feature[1] for feature in input_features], padding_value=0, max_length=max_sequence_length
            ),
            dtype=torch.long,
        )
        padded_token_type_ids = torch.tensor(
            pad_sequences(
                [feature[2] for feature in input_features], padding_value=0, max_length=max_sequence_length
            ),
            dtype=torch.long,
        )

        return (
            padded_token_ids,
            padded_attention_mask,
            padded_token_type_ids,
        )

# px = PAWSDataset()
self = px