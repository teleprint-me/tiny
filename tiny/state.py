"""
Module: tiny.state
Description: A super simple implementation for tracking the model state.

Do not store the optimizer or criterion within the model.
This way, they can be easily adjusted between runs.
Only relevant configuration parameters should be stored alongside the weights.
Common parameters between training and inference are considered relevant parameters.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer

from tiny.config import TinyConfig
from tiny.model import TinyTransformer
from tiny.tokenizer import TinyTokenizer


class TinyState:
    def __init__(self, model_path: str, model_config: dataclass = None):
        self.path = model_path
        self.config = model_config
        self.tokenizer = TinyTokenizer()
        self.checkpoint = None
        self.model = None

    def create_optimizer(self) -> Optimizer:
        """Return the AdamW optimizer."""
        return AdamW(
            self.model.parameters(recurse=self.config.recurse),
            lr=self.config.lr,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
            amsgrad=self.config.amsgrad,
        )

    def create_criterion(self) -> nn.Module:
        """Return the CrossEntropyLoss module."""
        return nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            reduction=self.config.reduction,
        )

    def create_checkpoint(self) -> dict[str, any]:
        """Return the model state dict."""
        try:
            return torch.load(self.model_path)
        except (FileNotFoundError,):
            return {}

    def load_model(self) -> TinyTransformer:
        self.checkpoint = self.create_checkpoint()

        # Do **not** override user configuration
        if not self.config and "config" in self.checkpoint:
            self.config = TinyConfig(**self.checkpoint["config"])

        # Create the model
        self.model = TinyTransformer(self.config)

        # Load the state dict if it exists
        if "state" in self.checkpoint:
            self.model.load_state_dict(self.checkpoint["state"])

        # Move the model to the available device
        self.model.to(self.config.device)

        return self.model

    def save_model(self) -> None:
        pass
