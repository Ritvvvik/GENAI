from pathlib import Path
import torch
from torch.utils.data import Dataset
from itertools import chain


class CustomDataset(Dataset):
    def __init__(self, tokenizer, data_dir, context_length):
        self.tokenizer = tokenizer
        texts = []
        for file in Path(data_dir).glob("**/*.txt"):
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            cleaned_lines = [line.strip() for line in lines]
            texts.append(cleaned_lines)
        # print(texts)

        token_ids = []
        for text in texts:
            encoded = self.tokenizer(text)
            flatten_ids = list(chain.from_iterable(encoded["input_ids"]))
            token_ids.extend(flatten_ids)
        # print(token_ids)

        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_ids) - context_length, context_length):
            input = token_ids[i : i + context_length]
            target = token_ids[i + 1 : i + context_length + 1]
            self.input_ids.append(torch.tensor(input))
            self.target_ids.append(torch.tensor(target))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
