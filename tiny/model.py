"""
Module: tiny.model
Description: This module contains a simplified decoder-only Transformer model for natural language processing.

This model avoids the use of dropout, which has been shown to negatively impact performance by randomly dropping
a selected percentage of samples from the propagated sequence. This can result in stuttering and repetition issues.
This model is inspired by Jamil's Transformer, Karpathy's GPT-2, and Meta's Llama architectures.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.tokens = nn.Embedding(vocab_size, d_model)
        self.encodings = PositionalEncoding(d_model, max_seq, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale by sqrt(d_model), Vaswani et al.’s embedding scaling trick
        scale = torch.sqrt(torch.tensor(self.d_model, dtype=x.dtype))
        tokens = self.tokens(x) * scale
        return self.encodings(tokens)


class MultiHeadSelfAttention(nn.Module):
    """Hybrid Causal Self-Attention block inspired by GPT-2 and Transformer models."""

    def __init__(self, d_model: int, num_heads: int, max_seq: int):
        super().__init__()
        assert d_model % num_heads == 0, "Embedding dim must be divisible by number of heads"

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads  # Per-head dimension

        # Linear projections for Q, K, V (each projects to d_model size)
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        # Final output projection
        self.wo = nn.Linear(d_model, d_model)

        # Precompute causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq, max_seq)).view(1, 1, max_seq, max_seq),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape  # Batch, Seq Len, Embedding Dim

        # Project Q, K, V
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, T, d_k)
        k = k.view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))
        attn_scores = attn_scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        y = attn_probs @ v  # (B, num_heads, T, d_k)

        # Reshape back to original size
        y = y.transpose(1, 2).contiguous().view(B, T, D)  # (B, T, D)

        # Final projection
        return self.wo(y)


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


class FeedForward(nn.Module):
    """Hybrid FeedForward block inspired by GPT-2 and Transformer models."""

    def __init__(self, d_model: int, ff_mult: float = 4.0):
        super().__init__()
        hidden = int(ff_mult * d_model)  # Hidden layer size
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.act = nn.GELU(approximate="tanh")  # GPT-2 uses GELU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))
