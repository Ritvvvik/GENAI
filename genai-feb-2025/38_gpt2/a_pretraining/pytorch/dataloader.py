import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data = load_from_disk(data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tmp = self.data[idx]["input_ids"]
        return torch.tensor(tmp[:-1]), torch.tensor(tmp[1:])
    

def create_data_loader(data_dir, batch_size, shuffle):
    print(">>> Begin Creating Dataloader...")
    dataset = CustomDataset(data_dir)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16
    )
    print("Total batches:", len(dataloader))
    print(">>> End Creating Dataloader...")
    return dataloader
    