"""
数据集加载策略
"""
import numpy as np
import torch
from torch.utils.data import Dataset

class UserDataset(Dataset):

    def __init__(self):
        # 输入数据
        self.user_seq = torch.tensor(np.load('data/user_seq.npy'), dtype=torch.int64)
        # [num_user, max_seq_len] 输入数据
        self.user_res = torch.tensor(np.load('data/user_res.npy'), dtype=torch.int64)
        # [num_user, max_seq_len] 输入标签
        self.user_mask = torch.tensor(np.load('data/user_mask.npy'), dtype=torch.bool)
        # [num_user, max_seq_len] 有值效记录
        idx2user = np.load('data/idx2user.npy', allow_pickle=True).item()
        keys_array = np.array(list(idx2user.values()))
        array_2d = np.repeat(keys_array[:, np.newaxis], 200, axis=1)
        self.id2userid = torch.tensor(array_2d)
    def __getitem__(self, index):
        # 返回[num_user, max_seq_len, 3]
        return torch.stack([self.user_seq[index], self.user_res[index], self.user_mask[index], self.id2userid[index]], dim=-1)

    def __len__(self):
        return self.user_seq.shape[0]