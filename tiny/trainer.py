"""
Copyright © 2025 Austin Berrio
Module: tiny.trainer
Description: A simple trainer for the Tiny Transformer model.

The state is responsible for keeping everything in sync, so the config is abdicated to the state.
I intentionally omit the scheduler to keep things as reasonably simple as possible.
Also, I prefer not using a scheduler and typically lean towards at a static learning rate.
"""

import torch
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

        self.dataset = None
        self.optimizer = None
        self.criterion = None

    # === 🔥 Convenience Properties === #
    @property
    def config(self) -> TinyConfig:
        return self.state.config

    @property
    def tokenizer(self) -> TinyTokenizer:
        return self.state.tokenizer

    @property
    def model(self) -> TinyTransformer:
        return self.state.model

    @property
    def device(self) -> TinyConfig:
        return self.config.device

    # === 🔥 Training Methods === #
    def create_optimizer(self) -> Optimizer:
        """Return the AdamW optimizer."""
        self.logger.info("Using optimizer: AdamW")
        return AdamW(
            self.model.parameters(recurse=self.config.recurse),
            lr=self.config.lr,
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
            amsgrad=self.config.amsgrad,
        )

    def create_criterion(self) -> nn.Module:
        """Return the CrossEntropyLoss module."""
        self.logger.info("Using criterion: Cross Entropy")
        return nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            reduction=self.config.reduction,
        )

    def train(self) -> None:
        self.logger.info("Starting training...")
        self.state.load_model()
        self.dataset = TinyDataset(self.config, self.tokenizer)
        self.state.config.max_seq = self.dataset.config.max_seq
        self.optimizer = self.create_optimizer()
        self.criterion = self.create_criterion()
        self.log_parameters()

        for epoch in range(self.config.num_epochs):
            total_loss = 0
            for batch, (x, y) in enumerate(self.dataset):
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / self.config.grad_accum_steps

                loss.backward()

                if (batch + 1) % self.config.grad_accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += loss.item() * self.config.grad_accum_steps
                self.log_batch(epoch, batch, loss)
            self.log_epoch(epoch, total_loss)

            save_every = (epoch + 1) % self.config.save_every == 0
            save_last = (epoch + 1) == self.config.num_epochs
            if save_every or save_last:
                self.state.save_model()

        self.logger.info("Training complete!")

    # === 🔥 Logging & Utilities === #
    def log_parameters(self):
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model has {num_params:,} learnable parameters.")

    def log_batch(self, epoch: int, batch: int, loss: torch.Tensor):
        """Logs loss & perplexity for each batch."""
        self.logger.debug(
            f"[Epoch: {epoch+1}/{self.config.num_epochs}] "
            f"[Batch: {batch}/{len(self.dataset) // self.config.batch_size}] "
            f"[Loss: {loss.item():.6f}] "
            f"[Perplexity: {self.perplexity(loss):.6f}]"
        )

    def log_epoch(self, epoch: int, total_loss: float):
        """Logs total epoch loss, learning rate, and perplexity."""
        average_loss = self.average_loss(total_loss)
        lr = self.optimizer.param_groups[0]["lr"]

        self.logger.info(
            f"[Epoch: {epoch+1}/{self.config.num_epochs}] "
            f"[Total Loss: {total_loss:.4f}] "
            f"[Avg Loss: {average_loss:.4f}] "
            f"[LR: {lr:.8f}] "
            f"[Perplexity: {self.perplexity(average_loss):.6f}]"
        )

    def average_loss(self, x: float | torch.Tensor) -> float:
        if isinstance(x, torch.Tensor):
            x = x.item()
        return x / len(self.dataset)

    def perplexity(self, x: float | torch.Tensor) -> float:
        """Computes perplexity, ensuring loss is non-negative."""
        if isinstance(x, float):
            x = torch.tensor(x)
        x = torch.clamp(x, min=0)
        return torch.exp(x).item()


if __name__ == "__main__":
    from tiny.args import TinyArgs

    args = TinyArgs("Tiny Trainer CLI").parse_args("trainer")

    config = TinyConfig(
        # General
        verbose=args.verbose,
        # Device
        seed=args.seed,
        dname=args.dname,
        dtype=torch.float32,
        # Tokenizer
        vocab_path=args.vocab_path,
        pad=args.pad_token,
        bos=args.bos_token,
        eos=args.eos_token,
        unk=args.unk_token,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
        vocab_size=149020,
        # Model
        model_path=args.model_path,
        max_seq=args.max_seq,
        d_model=args.d_model,
        num_heads=args.num_heads,
        eps=args.eps,
        ff_mult=args.ff_mult,
        num_layers=args.num_layers,
        # Dataset
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        shuffle=args.shuffle,  # TODO: Not implemented
        # Trainer
        num_epochs=args.num_epochs,
        save_every=args.save_every,
        grad_accum_steps=args.grad_accum_steps,
        # Optimizer
        recurse=args.recurse,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad,
        # Criterion
        reduction=args.reduction,
    )

    trainer = TinyTrainer(config)
    trainer.train()
