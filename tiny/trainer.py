"""
Module: tiny.trainer
Description: A simple trainer for the Tiny Transformer model.

The state is responsible for keeping everything in sync, so the config is abdicated to the state.
"""

import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer

from tiny.config import TinyConfig
from tiny.model import TinyTransformer
from tiny.state import TinyState
from tiny.tokenizer import TinyTokenizer


class TinyTrainer:
    def __init__(self, config: TinyConfig):
        self.state = TinyState(config)
        self.logger = config.logger(self.__class__.__name__, config.verbose)

    @property
    def config(self) -> TinyConfig:
        return self.state.config

    @property
    def tokenizer(self) -> TinyTokenizer:
        return self.state.tokenizer

    @property
    def model(self) -> TinyTransformer:
        return self.state.model

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
