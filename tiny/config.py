"""
Module: tiny.config
Description: User defined model and pipeline configuration settings.

This is a bit of a mess at the moment, but it's still a work in progress.
Some configurations are set dynamically:

- vocab_size
- max_seq

While others overlap:

- eps
- max_seq

So, it's fine for now since it simplifies the general pipeline.

Some parameters need to be overriden by the model state and are shared between
the trainer and generator. These special parameters are "frozen" at runtime.
"""

import logging
import random
import sys
from dataclasses import dataclass

import torch


@dataclass
class TinyConfig:
    # Device
    seed: int = 42
    dname: str = "cpu"
    dtype: torch.dtype = torch.float32

    # Tokenizer
    vocab_path: str = "data/vocab.json"
    pad: str = "<pad>"
    bos: str = "<bos>"
    eos: str = "<eos>"
    unk: str = "<unk>"
    add_bos: bool = True
    add_eos: bool = True
    vocab_size: int = 104

    # Dataset
    dataset_path: str = "data/tiny.json"
    max_seq: int = 128
    batch_size: int = 8
    shuffle: bool = False

    # Model
    model_path: str = "models/tiny.pth"
    d_model: int = 256
    num_heads: int = 16
    eps: float = 1e-6
    ff_mult: float = 4.0
    num_layers: int = 4

    # Trainer
    num_epochs: int = 10
    save_every: int = 10
    grad_accum_steps: int = 1
    # Optimizer
    recurse: bool = True
    lr: float = 1e-5
    weight_decay: float = 1e-2
    amsgrad: bool = False
    # Criterion
    reduction: str = "mean"

    # Generator
    # TODO

    # General
    verbose: bool = False

    def __post_init__(self):
        assert self.d_model % 2 == 0, "d_model must be even for sinusoidal encoding"
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        self.head_dim = self.d_model // self.num_heads
        self.scale = self.head_dim**-0.5
        self.hidden_dim = int(self.d_model * self.ff_mult)

        self._set_device()
        self._set_seed()

    @property
    def device(self) -> torch.device:
        """Returns the best available device as a `torch.device` object."""
        return torch.device(self.dname)

    def _set_device(self) -> None:
        """Sets the default device and dtype, and clears CUDA cache if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.dname = "cuda"
        if torch.mps.is_available():
            torch.mps.empty_cache()
            self.dname = "mps"
        torch.set_default_dtype(self.dtype)

    def _set_seed(self) -> None:
        """Sets the random seed for reproducibility."""
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    @property
    def _frozen_params(self) -> set[str]:
        """Parameters that are stored in the model checkpoint."""
        return {
            "pad",
            "bos",
            "eos",
            "unk",
            "add_bos",
            "add_eos",
            "max_seq",
            "d_model",
            "num_heads",
            "eps",
            "ff_mult",
            "num_layers",
        }

    def get_frozen_params(self) -> dict[str, any]:
        """Gets frozen parameters for a model checkpoint."""
        return {k: v for k, v in self.__dict__.items() if k in self._frozen_params}

    def set_frozen_params(self, state_dict: dict) -> None:
        """Sets frozen parameters from a model checkpoint."""
        for key, value in state_dict.items():
            if key in self._frozen_params:
                setattr(self, key, value)

    @classmethod
    def logger(cls, cls_name: str, verbose: bool) -> logging.Logger:
        """
        Initialize and return a Logger instance.

        :param cls_name: The name of the class that inherits from Logger.
        :param verbose: A boolean indicating whether to enable verbose logging.
        :return: Configured logger instance.
        """
        level = logging.DEBUG if verbose else logging.INFO
        fmt = "%(levelname)s:%(filename)s:%(lineno)d: %(message)s"
        logger = logging.getLogger(name=cls_name)
        logger.setLevel(level)

        # Check if the logger has handlers to avoid adding duplicates
        if not logger.hasHandlers():
            handler = logging.StreamHandler(stream=sys.stdout)
            formatter = logging.Formatter(fmt=fmt)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger
