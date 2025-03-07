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
        target = self.tokenizer.encode(item["output"])
        return {
            "input": torch.tensor(source, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
        }


# Example usage
if __name__ == "__main__":
    from tokenizer import Tokenizer

    tokenizer = Tokenizer()
    dataset = [
        {"input": "hello", "output": "world"},
        {"input": "how are you", "output": "i am fine"},
    ]

    ds = TinyDataset(dataset, tokenizer)
    print(ds[0])
