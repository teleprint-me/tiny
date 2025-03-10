"""
Module: tiny.trainer
Description: A simple trainer for the Tiny Transformer model.

The state is responsible for keeping everything in sync, so the config is abdicated to the state.
I intentionally omit the scheduler to keep things as reasonably simple as possible.
Also, I prefer not using a scheduler and typically lean towards at a static learning rate.
"""

import torch.nn as nn
from torch.optim.adamw import AdamW
from torch.optim.optimizer import Optimizer

from tiny.config import TinyConfig
from tiny.dataset import TinyDataset
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
    def dataset(self) -> TinyDataset:
        return self.state.dataset

    @property
    def model(self) -> TinyTransformer:
        return self.state.model

    @property
    def device(self) -> TinyConfig:
        return self.config.device

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

    def train(self) -> None:
        self.state.load_model()

        optimizer = self.create_optimizer()
        criterion = self.create_criterion()

        for epoch in range(self.config.num_epochs):
            total_loss = 0
            for batch, (x, y) in enumerate(self.dataset):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = criterion(logits.view(-1, logits.size(-1), y.view(-1)))
                loss = loss / self.config.grad_accum_steps

                loss.backward()

                if (batch + 1) % self.config.grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * self.config.grad_accum_steps

            if (epoch + 1) % self.config.save_every == 0:
                self.state.save_model()
