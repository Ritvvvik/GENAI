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
        ids = self.data[idx]["input_ids"][0]
        mask = self.data[idx]["attention_mask"][0]
        input_ids = ids[:-1]
        target_ids = ids[1:]
        selection_mask = mask[1:]
        for i, e in enumerate(selection_mask):
            if e == 0:
                target_ids[i] = -100
        return torch.tensor(input_ids), torch.tensor(target_ids)


def create_data_loader(data_dir, batch_size, shuffle):
    print(">>> Begin Creating Dataloader...")
    dataset = CustomDataset(data_dir)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count()
    )
    print("Total batches:", len(dataloader))
    print(">>> End Creating Dataloader...")
    return dataloader
    
if __name__ == "__main__":
    src_dir = "F:/nn/instruction-tuning/dolly"
    train_dir = os.path.join(src_dir, "tokenized_ds/train")
    dataset = CustomDataset(train_dir)
    for input_ids, target_ids in dataset:
        print(input_ids, target_ids)
