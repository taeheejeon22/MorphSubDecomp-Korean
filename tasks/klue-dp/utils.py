import numpy as np
import torch


class KlueDpInputExample:
    """
    A single training/test example for Dependency Parsing in .conllu format

    Args:
        guid : Unique id for the example
        text : string. the original form of sentence
        token_id : token id
        token : 어절
        pos : POS tag(s)
        head : dependency head
        dep : dependency relation
    """

    def __init__(
        self,
        guid: str,
        text: str,
        sent_id: int,
        token_id: int,
        token: str,
        pos: str,
        head: int,
        dep: str,
    ):
        self.guid = guid
        self.text = text
        self.sent_id = sent_id
        self.token_id = token_id
        self.token = token
        self.pos = pos
        self.head = head
        self.dep = dep


class KlueDpInputFeatures:
    """
    A single set of features of data. Property names are the same names as the corresponding inputs to a model.
    
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded)
            tokens.
        bpe_head_mask : Mask to mark the head token of bpe in aejeol
        head_ids : head ids for each aejeols on head token index
        dep_ids : dependecy relations for each aejeols on head token index
        pos_ids : pos tag for each aejeols on head token index
    """

    def __init__(
        self, guid, ids, mask, bpe_head_mask, bpe_tail_mask, head_ids, dep_ids, pos_ids
    ):
        self.guid = guid
        self.input_ids = ids
        self.attention_mask = mask
        self.bpe_head_mask = bpe_head_mask
        self.bpe_tail_mask = bpe_tail_mask
        self.head_ids = head_ids
        self.dep_ids = dep_ids
        self.pos_ids = pos_ids


def create_examples(file_path):
    sent_id = -1
    examples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "" or line == "\n" or line == "\t":
                continue

            if line.startswith("#"):
                parsed = line.strip().split("\t")
                if len(parsed) != 2:  # metadata line about dataset
                    continue
                else:
                    sent_id += 1
                    text = parsed[1].strip()
                    guid = parsed[0].replace("##", "").strip()
            else:
                token_list = []
                token_list = (
                    [sent_id]
                    + [token.replace("\n", "") for token in line.split("\t")]
                    + ["-", "-"]
                )
                examples.append(
                    KlueDpInputExample(
                        guid=guid,
                        text=text,
                        sent_id=sent_id,
                        token_id=int(token_list[1]),
                        token=token_list[2],
                        pos=token_list[4],
                        head=token_list[5],
                        dep=token_list[6],
                    )
                )
    return examples


def get_dp_labels():
    """
    label for dependency relations format:

    {structure}_(optional){function}

    """
    dp_labels = [
        "NP",
        "NP_AJT",
        "VP",
        "NP_SBJ",
        "VP_MOD",
        "NP_OBJ",
        "AP",
        "NP_CNJ",
        "NP_MOD",
        "VNP",
        "DP",
        "VP_AJT",
        "VNP_MOD",
        "NP_CMP",
        "VP_SBJ",
        "VP_CMP",
        "VP_OBJ",
        "VNP_CMP",
        "AP_MOD",
        "X_AJT",
        "VP_CNJ",
        "VNP_AJT",
        "IP",
        "X",
        "X_SBJ",
        "VNP_OBJ",
        "VNP_SBJ",
        "X_OBJ",
        "AP_AJT",
        "L",
        "X_MOD",
        "X_CNJ",
        "VNP_CNJ",
        "X_CMP",
        "AP_CMP",
        "AP_SBJ",
        "R",
        "NP_SVJ",
    ]
    return dp_labels


def get_pos_labels():
    """label for part-of-speech tags"""

    return [
        "NNG",
        "NNP",
        "NNB",
        "NP",
        "NR",
        "VV",
        "VA",
        "VX",
        "VCP",
        "VCN",
        "MMA",
        "MMD",
        "MMN",
        "MAG",
        "MAJ",
        "JC",
        "IC",
        "JKS",
        "JKC",
        "JKG",
        "JKO",
        "JKB",
        "JKV",
        "JKQ",
        "JX",
        "EP",
        "EF",
        "EC",
        "ETN",
        "ETM",
        "XPN",
        "XSN",
        "XSV",
        "XSA",
        "XR",
        "SF",
        "SP",
        "SS",
        "SE",
        "SO",
        "SL",
        "SH",
        "SW",
        "SN",
        "NA",
    ]


def flatten_prediction_and_labels(preds, labels):
    head_preds = list()
    head_labels = list()
    type_preds = list()
    type_labels = list()
    for pred, label in zip(preds, labels):
        head_preds += pred[0].cpu().flatten().tolist()
        head_labels += label[0].cpu().flatten().tolist()
        type_preds += pred[1].cpu().flatten().tolist()
        type_labels += label[1].cpu().flatten().tolist()
    head_preds = np.array(head_preds)
    head_labels = np.array(head_labels)
    type_preds = np.array(type_preds)
    type_labels = np.array(type_labels)

    index = [i for i, label in enumerate(head_labels) if label == -1]
    head_preds = np.delete(head_preds, index)
    head_labels = np.delete(head_labels, index)
    index = [i for i, label in enumerate(type_labels) if label == -1]
    type_preds = np.delete(type_preds, index)
    type_labels = np.delete(type_labels, index)

    return (
        head_preds.tolist(),
        type_preds.tolist(),
        head_labels.tolist(),
        type_labels.tolist(),
    )


def flatten_labels(labels):
    head_labels = list()
    type_labels = list()
    for label in labels:
        head_labels += label[0].cpu().flatten().tolist()
        type_labels += label[1].cpu().flatten().tolist()
    head_labels = np.array(head_labels)
    type_labels = np.array(type_labels)

    index = [i for i, label in enumerate(head_labels) if label == -1]
    head_labels = np.delete(head_labels, index)
    index = [i for i, label in enumerate(type_labels) if label == -1]
    type_labels = np.delete(type_labels, index)

    # classify others label as -3
    others_idx = 15
    for i, label in enumerate(type_labels):
        if label >= others_idx:
            type_labels[i] = -3

    return head_labels.tolist(), type_labels.tolist()


def resize_outputs(outputs, bpe_head_mask, bpe_tail_mask, max_word_length):
    batch_size, input_size, hidden_size = outputs.size()
    word_outputs = torch.zeros(batch_size, max_word_length + 1, hidden_size * 2).to(
        outputs.device
    )
    sent_len = list()

    for batch_id in range(batch_size):
        head_ids = [i for i, token in enumerate(bpe_head_mask[batch_id]) if token == 1]
        tail_ids = [i for i, token in enumerate(bpe_tail_mask[batch_id]) if token == 1]
        assert len(head_ids) == len(tail_ids)

        word_outputs[batch_id][0] = torch.cat(
            (outputs[batch_id][0], outputs[batch_id][0])
        )  # replace root with [CLS]
        for i, (head, tail) in enumerate(zip(head_ids, tail_ids)):
            word_outputs[batch_id][i + 1] = torch.cat(
                (outputs[batch_id][head], outputs[batch_id][tail])
            )
        sent_len.append(i + 2)

    return word_outputs, sent_len
