import torch
from dataset import KlueDpDataset
from torch.utils.data import DataLoader
from utils import get_pos_labels


class KlueDpDataLoader(object):
    def __init__(self, args, tokenizer, data_dir):
        self.args = args
        self.data_dir = data_dir
        self.dataset = KlueDpDataset(args, tokenizer)

    def collate_fn(self, batch):
        # 1. set args
        batch_size = len(batch)
        pos_padding_idx = None if self.args.no_pos else len(get_pos_labels())
        # 2. build inputs : input_ids, attention_mask, bpe_head_mask, bpe_tail_mask
        input_ids = []
        attention_masks = []
        bpe_head_masks = []
        bpe_tail_masks = []
        for batch_id in range(batch_size):
            (
                input_id,
                attention_mask,
                bpe_head_mask,
                bpe_tail_mask,
                _,
                _,
                _,
            ) = batch[batch_id]
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            bpe_head_masks.append(bpe_head_mask)
            bpe_tail_masks.append(bpe_tail_mask)
        # 2. build inputs : packing tensors
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        bpe_head_masks = torch.stack(bpe_head_masks)
        bpe_tail_masks = torch.stack(bpe_tail_masks)
        # 3. token_to_words : set in-batch max_word_length
        max_word_length = max(torch.sum(bpe_head_masks, dim=1)).item()
        # 3. token_to_words : placeholders
        head_ids = torch.zeros(batch_size, max_word_length).long()
        type_ids = torch.zeros(batch_size, max_word_length).long()
        pos_ids = torch.zeros(batch_size, max_word_length + 1).long()
        mask_e = torch.zeros(batch_size, max_word_length + 1).long()
        # 3. token_to_words : head_ids, type_ids, pos_ids, mask_e, mask_d
        for batch_id in range(batch_size):
            (
                _,
                _,
                bpe_head_mask,
                _,
                token_head_ids,
                token_type_ids,
                token_pos_ids,
            ) = batch[batch_id]
            head_id = [i for i, token in enumerate(bpe_head_mask) if token == 1]
            word_length = len(head_id)
            head_id.extend([0] * (max_word_length - word_length))
            head_ids[batch_id] = token_head_ids[head_id]
            type_ids[batch_id] = token_type_ids[head_id]
            if not self.args.no_pos:
                pos_ids[batch_id][0] = pos_padding_idx
                pos_ids[batch_id][1:] = token_pos_ids[head_id]
                pos_ids[batch_id][torch.sum(bpe_head_mask) + 1 :] = pos_padding_idx
            mask_e[batch_id] = torch.LongTensor(
                [1] * (word_length + 1) + [0] * (max_word_length - word_length)
            )
        mask_d = mask_e[:, 1:]
        # 4. pack everything
        masks = (attention_masks, bpe_head_masks, bpe_tail_masks, mask_e, mask_d)
        ids = (head_ids, type_ids, pos_ids)

        return input_ids, masks, ids, max_word_length

    def get_test_dataloader(self, data_filename: str = "klue-dp-v1_test.tsv", **kwargs):
        dataset = self.dataset.get_test_dataset(self.data_dir, data_filename)
        return DataLoader(
            dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            **kwargs
        )
