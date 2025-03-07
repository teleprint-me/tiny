"""
Module: tiny.dataset
Description: Simple dataset wrapper for training.
"""

import torch
from torch.utils.data import Dataset

from tiny.tokenizer import TinyTokenizer


class TinyProcessor:
    def __init__(self, tokenizer: TinyTokenizer, dataset: list[dict[str, str]]):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_seq = 0

    def calc_max_seq(self):
        # Calculate the longest pair
        max_seq = 0
        for pair in self.dataset:
            # input is the beginning of the sequence
            source = tokenizer.encode(pair["input"], add_bos=True)
            # target is the predicted output which continues on from the input sequence
            target = tokenizer.encode(pair["target"], add_eos=True)
            # calculate the max length of any given sequence
            max_seq = max(len(source) + len(target), max_seq)
        self.max_seq = max_seq

    def pad(self, tokens: list[int]) -> list[int]:
        # append a sequence of pad ids to a given sequence up to its max length
        return tokens + [self.tokenizer.pad_id] * (self.max_seq - len(tokens))

    def encode(self) -> list[dict[str, list[int]]]:
        # only do this right before we process and encode the input-target pairs
        self.calc_max_seq()

        # preprocess the input-target pairs
        encoded = []
        for pair in self.dataset:
            source = self.pad(tokenizer.encode(pair["input"], add_bos=True))
            target = self.pad(tokenizer.encode(pair["target"], add_eos=True))
            encoded.append({"input": source, "target": target})
        return encoded


class TinyDataset(Dataset):
    def __init__(self, tokenizer: TinyTokenizer, dataset: list[dict[str, str]]):
        # preprocess the dataset
        self.processor = TinyProcessor(tokenizer, dataset)
        # encode and pad the dataset for training
        self.dataset = self.processor.encode()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input": torch.tensor(item["input"], dtype=torch.long),
            "target": torch.tensor(item["target"], dtype=torch.long),
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
