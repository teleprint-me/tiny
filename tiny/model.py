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


class PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding for transformer models."""

    def __init__(self, d_model: int, max_seq: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sinusoidal encoding"
        self.d_model = d_model
        self.max_seq = max_seq
        self.dtype = dtype

        # Register precomputed positional encodings as a non-trainable buffer
        pe = self._generate_sinusoidal_encoding()
        self.register_buffer("pe", pe, persistent=False)  # Explicitly non-trainable

    def _generate_sinusoidal_encoding(self) -> torch.Tensor:
        """Creates sinusoidal positional encodings as described in Vaswani et al. (2017)."""
        pe = torch.zeros(self.max_seq, self.d_model, dtype=self.dtype)
        position = torch.arange(0, self.max_seq, dtype=self.dtype).unsqueeze(1)
        step = torch.arange(0, self.d_model, 2, dtype=self.dtype)
        scale = -torch.log(torch.tensor(10000.0)) / self.d_model
        div_term = torch.exp(step * scale)
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        return pe.unsqueeze(0)  # Shape: (1, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds positional encoding to the input tensor x."""
        return x + self.pe[:, : x.size(1), :]  # (B, T, D)


class PositionalEmbedding(nn.Module):
    """Token + positional embedding for transformer decoder."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq: int,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoding = PositionalEncoding(d_model, max_seq, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale by sqrt(d_model), Vaswani et al.’s embedding scaling trick
        scale = torch.sqrt(torch.tensor(self.d_model, dtype=x.dtype))
        embeddings = self.embedding(x) * scale
        return self.encoding(embeddings)


class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))  # Learnable scale
        self.bias = nn.Parameter(torch.zeros(d_model))  # Learnable bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)  # More stable than std
        return self.alpha * (x - mean) / torch.sqrt(var + self.eps) + self.bias
