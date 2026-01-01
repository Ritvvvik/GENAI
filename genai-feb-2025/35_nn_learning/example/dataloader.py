import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, file):
        self.df = pd.read_csv(file)
        X = np.array(self.df.x, dtype=np.float32).reshape(-1, 1)
        self.X = torch.Tensor(X)
        y = np.array(self.df.y, dtype=np.float32).reshape(-1, 1)
        self.y = torch.Tensor(y)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_loader(dir, file_name, batch_size, shuffle):
    # create an iterable train and test datasets
    dataset = CustomDataset(os.path.join(dir, file_name))

    # create a batch-iterable train and test datasets
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
