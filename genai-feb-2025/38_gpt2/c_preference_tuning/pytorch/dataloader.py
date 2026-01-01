import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk

class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data = load_from_disk(data_dir)

    def __len__(self):
        return len(self.data)

    def helper(self, ids, mask):
        input_ids = ids[:-1]
        target_ids = ids[1:]
        selection_mask = mask[1:]
        for i, e in enumerate(selection_mask):
            if e == 0:
                target_ids[i] = -100
        return torch.tensor(input_ids), torch.tensor(target_ids)
    
    def __getitem__(self, idx):
        chosen_ids = self.data[idx]["chosen_input_ids"][0]
        choosen_mask = self.data[idx]["chosen_attention_mask"][0]
        chosen_ids, chosen_target_ids = self.helper(chosen_ids, choosen_mask)

        rej_ids = self.data[idx]["rej_input_ids"][0]
        rej_mask = self.data[idx]["rej_attention_mask"][0]
        rej_ids, rej_target_ids = self.helper(rej_ids, rej_mask)
        return chosen_ids, chosen_target_ids, rej_ids, rej_target_ids
    

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
    src_dir = "F:/nn/preference_tuning/orca"
    train_dir = os.path.join(src_dir, "tokenized_ds/train")
    dataset = CustomDataset(train_dir)
    for chosen_input_ids, chosen_target_ids, rej_input_ids, rej_target_ids in dataset:
        print(chosen_input_ids, chosen_target_ids)
        print(rej_input_ids, rej_target_ids)
        break
