import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


class CustomDataset(Dataset):
    def __init__(self, file):
        self.wine_df = pd.read_csv(file)
        X, y = self.preprocess(self.wine_df)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def preprocess(self, df):
        features = []
        for col in df.columns:
            if col != "quality":
                features.append(col)
        scaler = StandardScaler()
        X = scaler.fit_transform(df[features])
        y = np.array(df.quality).reshape(-1, 1)
        return X, y

    def __len__(self):
        return self.wine_df.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_loader(dir, file_name, batch_size, shuffle):
    # create an iterable train and test datasets
    dataset = CustomDataset(os.path.join(dir, file_name))

    # create a batch-iterable train and test datasets
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    src_dir = "C:/Users/pc/Documents/nn/regression"
    train_loader = get_loader(src_dir, "train.csv", 10, True)
    for item in train_loader:
        print(item)
        break
