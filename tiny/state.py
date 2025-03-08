"""
Module: tiny.state
Description: A super simple implementation for tracking the model state.

Do not store the optimizer or criterion within the model.
This way, they can be easily adjusted between runs.
Only relevant configuration parameters should be stored alongside the weights.
Common parameters between training and inference are considered relevant parameters.
"""

import torch
import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer

from tiny.model import TinyTransformer
from tiny.tokenizer import TinyTokenizer


class TinyState:
    tokenizer: TinyTokenizer
    transformer: TinyTransformer

    def create_optimizer(self) -> Optimizer:
        """Return the AdamW optimizer."""
        return AdamW(
            self.transformer.parameters(recurse=self.config.recurse),
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
