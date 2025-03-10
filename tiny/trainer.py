"""
Module: tiny.trainer
Description: A simple trainer for the Tiny Transformer model.
"""

import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer

from tiny.config import TinyConfig
from tiny.model import TinyTransformer
from tiny.state import TinyState
from tiny.tokenizer import TinyTokenizer


class TinyTrainer:
    def __init__(self, state: TinyState):
        self._state = state

    @property
    def config(self) -> TinyConfig:
        return self._state.config

    @property
    def state(self) -> TinyState:
        return self._state

    @property
    def tokenizer(self) -> TinyTokenizer:
        return self._state.tokenizer

    @property
    def model(self) -> TinyTransformer:
        return self._state.model

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
