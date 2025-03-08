"""
Module: tiny.state
Description: A super simple implementation for tracking the model state.

Do not store the optimizer or criterion within the model.
This way, they can be easily adjusted between runs.
Only relevant configuration parameters should be stored alongside the weights.
Common parameters between training and inference are considered relevant parameters.
"""

import json

import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer

from tiny.config import TinyConfig
from tiny.dataset import TinyDataset
from tiny.model import TinyTransformer
from tiny.tokenizer import TinyTokenizer


class TinyState:
    def __init__(self, model_path: str, dataset_path: str, **kwargs):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.config = kwargs

        self.dataset = self.create_dataset()
        self.tokenizer = self.create_tokenizer()
        self.checkpoint = self.create_checkpoint()
        self.model = self.load_model()

    def create_tokenizer(self) -> TinyTokenizer:
        tokenizer = TinyTokenizer(
            pad=self.config.get("pad", "<pad>"),
            bos=self.config.get("bos", "<bos>"),
            eos=self.config.get("eos", "<eos>"),
            unk=self.config.get("unk", "<unk>"),
        )
        return tokenizer

    def create_dataset(self) -> TinyDataset:
        dataset = None
        with open(self.dataset_path, "r") as file:
            dataset = TinyDataset(
                tokenizer=self.tokenizer,
                dataset=json.load(file),
                max_seq=self.config.get("max_seq", 128),
                batch_size=self.config.get("batch_size", 8),
            )
        return dataset

    def create_optimizer(self) -> Optimizer:
        """Return the AdamW optimizer."""
        return AdamW(
            self.model.parameters(recurse=self.config.recurse),
            lr=self.config.get("lr", 1e-5),
            eps=self.config.get("eps", 1e-5),
            weight_decay=self.config.get("weight_decay", 1e-2),
            amsgrad=self.config.get("amsgrad", False),
        )

    def create_criterion(self) -> nn.Module:
        """Return the CrossEntropyLoss module."""
        return nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            reduction=self.config.get("reduction", "mean"),
        )

    def create_checkpoint(self) -> dict[str, any]:
        """Return the model state dict."""
        try:
            return torch.load(self.model_path)
        except (FileNotFoundError,):
            return {}

    def create_model(self) -> TinyTransformer:
        return TinyTransformer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=self.config.get("d_model", 256),
            num_layers=self.config.get("num_layers", 4),
            num_heads=self.config.get("num_heads", 16),
            max_seq=self.dataset.max_seq,
        )

    def load_model(self) -> TinyTransformer:
        self.checkpoint = self.create_checkpoint()

        # Do **not** override user configuration
        if not self.config and "config" in self.checkpoint:
            self.config = self.checkpoint.get("config", {})

        # Create the model
        self.model = self.create_model()

        # Load the state dict if it exists
        if "state" in self.checkpoint:
            self.model.load_state_dict(self.checkpoint.get("state", {}))

        # Move the model to the available device
        self.model.to(self.config.device)

        return self.model

    def save_model(self) -> None:
        pass
