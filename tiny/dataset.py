"""
Module: tiny.dataset
Description: Simple dataset wrapper for training.
"""

import torch
from torch.utils.data import Dataset

from tiny.tokenizer import TinyTokenizer


class TinyProcessor:
    def __init__(
        self,
        tokenizer: TinyTokenizer,
        dataset: list[dict[str, str]],
        max_seq: int = 128,
    ):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_seq = max_seq

    def calc_max_seq(self):
        # Calculate the longest pair
        for pair in self.dataset:
            # input is the beginning of the sequence
            source = tokenizer.encode(pair["input"], add_bos=True)
            # target is the predicted output which continues on from the input sequence
            target = tokenizer.encode(pair["target"], add_eos=True)
            # calculate the max length of any given sequence
            self.max_seq = max(len(source) + len(target), self.max_seq)
        # Max seq len must be evenly divisible
        if self.max_seq % 2 != 0:
            self.max_seq += 1  # Should probably be some power of 2, but I'm being lazy

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
    def __init__(
        self,
        tokenizer: TinyTokenizer,
        dataset: list[dict[str, str]],
        max_seq: int = 128,
        batch_size: int = 8,
    ):
        self.tokenizer = tokenizer
        processor = TinyProcessor(tokenizer, dataset, max_seq)
        self.dataset = processor.encode()
        self.max_seq = processor.max_seq
        self.batch_size = batch_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.dataset[idx]
        source = torch.tensor(pair["input"], dtype=torch.long)
        target = torch.tensor(pair["target"], dtype=torch.long)
        return source, target

    def __iter__(self) -> tuple[torch.Tensor, torch.Tensor]:
        for i in range(0, len(self), self.batch_size):
            batch = self.dataset[i : i + self.batch_size]
            shape = (len(batch), self.max_seq)
            source = torch.full(shape, self.tokenizer.pad_id, dtype=torch.long)
            target = torch.full(shape, self.tokenizer.pad_id, dtype=torch.long)
            for j, pair in enumerate(batch):
                source[j] = torch.tensor(pair["input"], dtype=torch.long)
                target[j] = torch.tensor(pair["target"], dtype=torch.long)
            yield (source, target)


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
    x, y = dataset[0]
    print(f"Max seq: {dataset.max_seq}")
    print(f"Input: {x}")
    print(f"Target: {y}")
