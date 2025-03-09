"""
Module: tiny.config
Description: User defined model and pipeline configuration settings.
"""

import random
from dataclasses import dataclass

import torch


@dataclass
class TinyConfig:
    # Device
    seed: int = 42
    dname: str = "cpu"
    dtype: torch.dtype = torch.float32

    # Tokenizer
    pad: str = "<pad>"
    bos: str = "<bos>"
    eos: str = "<eos>"
    unk: str = "<unk>"
    add_bos: bool = True
    add_eos: bool = True

    # Dataset
    dataset_path: str = "data/tiny.json"
    max_seq: int = 128
    batch_size: int = 8

    # Model
    model_path: str = "models/tiny.pth"
    d_model: int = 256
    num_heads: int = 16
    eps: float = 1e-6
    ff_mult: float = 4.0
    num_layers: int = 4

    # Optimizer
    recurse: bool = True
    lr: float = 1e-5
    weight_decay: float = 1e-2
    amsgrad: bool = False

    # Criterion
    reduction: str = "mean"

    # Trainer
    # TODO

    # Generator
    # TODO

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
    def _blacklist(self) -> set[str]:
        return {
            "seed",  # Device
            "dname",
            "dtype",
            "device",
            "dataset_path",
            "model_path",
            "head_dim",
            "hidden_dim",
            "scale",  # Computed at runtime
        }

    def as_dict(self) -> dict[str, any]:
        """Returns a dictionary representation of the config."""
        kv = {}
        for k, v in self.__dict__.items():
            if k not in self._blacklist and not k.startswith("_"):
                kv[k] = v
        return kv
