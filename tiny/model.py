"""
Module: tiny.model
Description: A super simple decoder-only transformer implementation for natural language processing.

---

Do **not** use dropout. This is known to harm the model.
If we use a 0.1 dropout, that means 10% of samples are randomly dropped from the propogated sequence. This can cause stuttering and repetition issues. I omit dropout from the implementation as a result.

This will be a mix of Jamils Transformer, Karpathys GPT-2, and possibly some aspects of Metas Llama.
"""

import torch
from torch import nn


class PositionalEncoding:
    """Standard fixed sinusoidal positional encoding for transformer models."""

    def __init__(self, d_model: int, max_seq: int, dtype: torch.dtype):
        super().__init__()
        self.d_model = d_model
        self.max_seq = max_seq
        self.dtype = dtype

        # Register precomputed positional encodings as a non-trainable buffer
        self.register_buffer("pe", self._generate_sinusoidal_encoding())

    # ref: https://stackoverflow.com/a/77445896/20035933
    def _generate_sinusoidal_encoding(self) -> torch.Tensor:
        """Creates sinusoidal positional encodings as described in Vaswani et al. (2017)."""
        # Create a tensor of shape (max_seq, d_model) to store the positional encodings.
        pe = torch.zeros(self.max_seq, self.d_model)
        # Create a tensor of shape (max_seq, 1) to store the position indices.
        position = torch.arange(0, self.max_seq, dtype=self.dtype).unsqueeze(1)
        # Calculate the division term of shape (D/2,) to avoid division by zero.
        d = torch.arange(0, self.d_model, 2, dtype=self.dtype)
        t = -torch.log(torch.tensor(10000.0)) / self.d_model
        div_term = torch.exp(d * t)
        # Compute the positional encoding for even and odd indices separately.
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(position * (10000 ** (2i / d_model))
        return pe.unsqueeze(0)  # (1, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor x."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]  # (B, T, D)


class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("eps", eps)
        self.alpha = nn.Parameter(torch.ones(d_model))  # Learnable scale
        self.bias = nn.Parameter(torch.zeros(d_model))  # Learnable bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)  # More stable than std
        return self.alpha * (x - mean) / torch.sqrt(var + self.eps) + self.bias
