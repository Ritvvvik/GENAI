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


# create an iterable train and test datasets
src_dir = "C:/Users/pc/Documents/nn/linear"
train_dataset = CustomDataset(os.path.join(src_dir, "train.csv"))
test_dataset = CustomDataset(os.path.join(src_dir, "test.csv"))

# create a batch-iterable train and test datasets
train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False)

if __name__ == "__main__":
    for item in train_dataset:
        print(item)
        break

    for item in test_dataset:
        print(item)
        break

    for item in train_loader:
        print(item)
        break

    for item in test_loader:
        print(item)
        break
