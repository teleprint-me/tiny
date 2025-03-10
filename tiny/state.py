"""
Module: tiny.state
Description: A super simple implementation for tracking the model state.

Do not store the optimizer or criterion within the model.
This way, they can be easily adjusted between runs.
Only relevant configuration parameters are stored alongside the weights.
Common parameters between training and inference are considered relevant parameters.
"""

import torch

from tiny.config import TinyConfig
from tiny.dataset import TinyDataset
from tiny.model import TinyTransformer
from tiny.tokenizer import TinyTokenizer


class TinyState:
    def __init__(self, config: TinyConfig):
        self.config = config
        self.logger = config.logger(self.__class__.__name__, config.verbose)

        self.tokenizer = None
        self.dataset = None
        self.checkpoint = None
        self.model = None

    def load_tokenizer(self) -> None:
        self.tokenizer = TinyTokenizer(self.config)
        self.config.vocab_size = self.tokenizer.vocab_size

    def load_dataset(self) -> None:
        self.dataset = TinyDataset(self.config, self.tokenizer)
        self.config.max_seq = self.dataset.config.max_seq

    def load_checkpoint(self) -> None:
        """Return the model state dict."""
        try:
            self.logger.info(f"Loading checkpoint from {self.config.model_path}.")
            self.checkpoint = torch.load(self.config.model_path)
        except (FileNotFoundError,):
            self.logger.info(f"No checkpoint found. Creating new state at {self.config.model_path}")
            self.checkpoint = {}

    def load_model(self) -> None:
        self.logger.debug("Loading model state...")

        # Load dependencies
        self.load_tokenizer()
        self.load_dataset()
        self.load_checkpoint()

        # Override select parameters between runs
        if "config" in self.checkpoint:
            self.config.set_frozen_params(self.checkpoint["config"])
            self.logger.info("Model config loaded from checkpoint.")

        # Create the model
        self.model = TinyTransformer(self.config)
        self.logger.info("Model created: TinyTransformer")

        # Load the state dict if it exists
        if "state" in self.checkpoint:
            self.model.load_state_dict(self.checkpoint["state"])
            self.logger.info("Model state loaded from checkpoint.")

        # Move the model to the available device
        self.model.to(self.config.device)
        self.logger.info(f"Model moved to device: {self.config.dname}")

    def save_model(self) -> None:
        """Save the model state along with relevant configuration."""
        self.logger.debug("Saving model state...")

        if self.model is None:
            raise RuntimeError("Model is not initialized, cannot save checkpoint.")

        state_dict = {
            "config": self.config.get_frozen_params(),
            "state": self.model.state_dict(),
        }
        torch.save(state_dict, self.config.model_path)
        self.logger.info(f"Model saved to {self.config.model_path}")
