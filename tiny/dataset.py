"""
Module: tiny.dataset
Description: Simple dataset wrapper for training.
"""

import torch
from torch.utils.data import Dataset

from tiny.tokenizer import TinyTokenizer


class TinyDataset(Dataset):
    def __init__(self, tokenizer: TinyTokenizer, dataset: list[dict[str, str]]):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        source = self.tokenizer.encode(item["input"])
        target = self.tokenizer.encode(item["target"])
        return {
            "input": torch.tensor(source, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
        }


# Example usage
if __name__ == "__main__":
    import json
    import os

    if os.path.exists("data/tiny.json"):
        with open("data/tiny.json", "r") as file:
            hotpot = json.load(file)
    else:
        hotpot = [
            {"input": "hello", "target": "world"},
            {"input": "how are you", "target": "i am fine"},
        ]

    tokenizer = TinyTokenizer()

    dataset = TinyDataset(tokenizer, hotpot)
    print(dataset[0])
