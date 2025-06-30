import os
import numpy as np
import torch
from torch.utils.data import Dataset

class C4IndexedDataset(Dataset):
    """
    读取由 SentencePiece 或类似工具预先生成的
    - c4_text_document.bin : dtype=uint16 的 token id 序列拼接
    - c4_text_document.idx : dtype=int32 的每条记录在 .bin 中的起始位置
    """
    def __init__(self, bin_path, idx_path, seq_len, pad_id=0):
        super().__init__()
        assert os.path.isfile(bin_path), f"{bin_path} not found"
        assert os.path.isfile(idx_path), f"{idx_path} not found"
        # memmap idx: 每个 entry 是样本起始 token 下标
        self.idx = np.memmap(idx_path, dtype=np.int64, mode='r', offset=4)
        # memmap bin: 连续的 uint16 token ids
        self.bin = np.memmap(bin_path, dtype=np.uint16, mode='r')
        # 样本数 = len(idx)-1
        self.n_samples = len(self.idx) - 1
        self.seq_len = seq_len
        self.pad_id = pad_id

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        # 取第 i 条样本的 token id slice
        start = int(self.idx[i])
        end = int(self.idx[i + 1])
        tokens = self.bin[start:end]
        # 如果样本长度 >= seq_len+1，就截断；否则 pad 到 seq_len+1
        if len(tokens) >= self.seq_len + 1:
            tokens = tokens[: self.seq_len + 1]
        else:
            pad_length = self.seq_len + 1 - len(tokens)
            tokens = np.concatenate([tokens, np.full(pad_length, self.pad_id, dtype=np.uint16)])
        # X = 前 seq_len, Y = 后 seq_len (下一个 token)
        x = torch.from_numpy(tokens[: self.seq_len]).long()
        y = torch.from_numpy(tokens[1 : self.seq_len + 1]).long()
        return x, y
