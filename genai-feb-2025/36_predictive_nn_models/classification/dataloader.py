import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


class CustomDataset(Dataset):
    def __init__(self, file):
        self.credit_df = pd.read_csv(file)
        X, y = self.preprocess(self.credit_df)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def preprocess(self, df):
        # preprocess categorical features
        cat_features = ["Payment_of_Min_Amount", "Credit_Mix", "Payment_Behaviour"]
        ohe = preprocessing.OneHotEncoder(sparse_output=False)
        cat_data = ohe.fit_transform(df[cat_features])
        print(cat_data.shape)

        # preprocess continuous features
        cont_features = []
        for col in df.columns:
            if col != "Credit_Score" and col not in cat_features:
                cont_features.append(col)
        scaler = StandardScaler()
        cont_data = scaler.fit_transform(df[cont_features])
        print(cont_data.shape)
        X = np.concat([cat_data, cont_data], axis=1)
        print(X.shape)

        # process target feature
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(df["Credit_Score"])
        y = y.reshape(-1, 1)
        return X, y

    def __len__(self):
        return self.credit_df.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_loader(dir, file_name, batch_size, shuffle):
    # create an iterable train and test datasets
    dataset = CustomDataset(os.path.join(dir, file_name))

    # create a batch-iterable train and test datasets
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    src_dir = "C:/Users/pc/Documents/nn/classification"
    train_loader = get_loader(src_dir, "train.csv", 10, True)
    for item in train_loader:
        print(item)
        break
