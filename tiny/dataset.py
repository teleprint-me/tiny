"""
Module: tiny.dataset
Description: Simple dataset wrapper for training.
"""

import json

import torch
from torch.utils.data import Dataset

from tiny.config import TinyConfig
from tiny.tokenizer import TinyTokenizer


class TinyProcessor:
    def __init__(self, config: TinyConfig, tokenizer: TinyTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = None
        self.logger = config.logger(self.__class__.__name__, config.verbose)

    def _load_dataset(self) -> None:
        self.logger.info(f"Loading dataset from {self.config.dataset_path}.")
        with open(self.config.dataset_path, "r") as file:
            self.dataset = json.load(file)
        self.logger.info("Successfully loaded dataset.")

    def _calc_max_seq(self) -> None:
        self.logger.info("Calculating the maximum sequence length.")
        # Calculate the longest pair
        for pair in self.dataset:
            # input is the beginning of the sequence
            source = self.tokenizer.encode(pair["input"], add_bos=self.config.add_bos)
            # target is the predicted output which continues on from the input sequence
            target = self.tokenizer.encode(pair["target"], add_eos=self.config.add_eos)
            # calculate the max length of any given sequence
            self.config.max_seq = max(len(source) + len(target), self.config.max_seq)

        # Max seq len must be evenly divisible
        if self.config.max_seq % 2 != 0:
            self.config.max_seq += 1  # Should probably be some power of 2, but I'm being lazy

        self.logger.info(f"Maximum sequence length set to {self.config.max_seq}")

    def _pad(self, tokens: list[int]) -> list[int]:
        # append a sequence of pad ids to a given sequence up to its max length
        return tokens + [self.tokenizer.pad_id] * (self.config.max_seq - len(tokens))

    def encode(self) -> list[dict[str, list[int]]]:
        self._load_dataset()
        self._calc_max_seq()

        self.logger.info(
            f"Encoding dataset: add_bos={self.config.add_bos}, add_eos={self.config.add_eos}"
        )

        # preprocess the input-target pairs
        encoded = []
        for pair in self.dataset:
            source = self._pad(self.tokenizer.encode(pair["input"], add_bos=self.config.add_bos))
            target = self._pad(self.tokenizer.encode(pair["target"], add_eos=self.config.add_eos))
            encoded.append({"input": source, "target": target})

        self.logger.info(f"Successfully encoded dataset using {len(self.dataset)} pairs")
        return encoded


class TinyDataset(Dataset):
    def __init__(self, config: TinyConfig, tokenizer: TinyTokenizer):
        self.config = config
        self.tokenizer = tokenizer

        self.processor = TinyProcessor(config, tokenizer)
        self.dataset = self.processor.encode()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        pair = self.dataset[idx]
        source = torch.tensor(pair["input"], dtype=torch.long)
        target = torch.tensor(pair["target"], dtype=torch.long)
        return source, target

    def __iter__(self) -> tuple[torch.Tensor, torch.Tensor]:
        for i in range(0, len(self), self.config.batch_size):
            batch = self.dataset[i : i + self.config.batch_size]
            shape = (len(batch), self.config.max_seq)
            source = torch.full(shape, self.tokenizer.pad_id, dtype=torch.long)
            target = torch.full(shape, self.tokenizer.pad_id, dtype=torch.long)
            for j, pair in enumerate(batch):
                source[j] = torch.tensor(pair["input"], dtype=torch.long)
                target[j] = torch.tensor(pair["target"], dtype=torch.long)
            yield (source, target)


# Example usage
if __name__ == "__main__":
    config = TinyConfig()
    tokenizer = TinyTokenizer(config)
    dataset = TinyDataset(config, tokenizer)

    x, y = dataset[0]
    print(f"Max seq: {dataset.config.max_seq}")
    print(f"Input: {x}")
    print(f"Target: {y}")
