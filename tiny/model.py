"""
Copyright © 2025 Austin Berrio
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

from tiny.config import TinyConfig


class PositionalEncoding(nn.Module):
    """Standard fixed sinusoidal positional encoding for transformer models."""

    def __init__(self, config: TinyConfig):
        super().__init__()
        self.d_model = config.d_model
        self.max_seq = config.max_seq
        self.dtype = config.dtype

        # Register precomputed positional encodings as a non-trainable buffer
        self.register_buffer("pe", self._generate_sinusoidal_encoding())

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

    def __init__(self, config: TinyConfig):
        super().__init__()
        self.d_model = config.d_model
        self.tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.encodings = PositionalEncoding(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale by sqrt(d_model), Vaswani et al.’s embedding scaling trick
        scale = torch.sqrt(torch.tensor(self.d_model, dtype=x.dtype))
        tokens = self.tokens(x) * scale
        return self.encodings(tokens)


class MultiHeadSelfAttention(nn.Module):
    """Hybrid Causal Self-Attention block inspired by GPT-2 and Transformer models."""

    def __init__(self, config: TinyConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        self.d_k = config.head_dim  # Per-head dimension

        # Linear projections for Q, K, V (each projects to d_model size)
        self.wq = nn.Linear(config.d_model, config.d_model)
        self.wk = nn.Linear(config.d_model, config.d_model)
        self.wv = nn.Linear(config.d_model, config.d_model)

        # Final output projection
        self.wo = nn.Linear(config.d_model, config.d_model)

        # Precompute causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.max_seq, config.max_seq)).view(
                1, 1, config.max_seq, config.max_seq
            ),
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
    def __init__(self, config: TinyConfig):
        super().__init__()
        self.eps = config.eps
        self.alpha = nn.Parameter(torch.ones(config.d_model))  # Learnable scale
        self.bias = nn.Parameter(torch.zeros(config.d_model))  # Learnable bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)  # More stable than std
        return self.alpha * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class FeedForward(nn.Module):
    """Hybrid FeedForward block inspired by GPT-2 and Transformer models."""

    def __init__(self, config: TinyConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.d_model)
        self.act = nn.GELU(approximate="tanh")  # GPT-2 uses GELU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DecoderBlock(nn.Module):
    """A single decoder block with self-attention and feedforward layers."""

    def __init__(self, config: TinyConfig):
        super().__init__()
        self.attn = MultiHeadSelfAttention(config)
        self.ffn = FeedForward(config)

        self.norm1 = LayerNormalization(config)
        self.norm2 = LayerNormalization(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention + Residual + LayerNorm
        x = x + self.attn(self.norm1(x))
        # Feedforward + Residual + LayerNorm
        x = x + self.ffn(self.norm2(x))
        return x


class TinyTransformer(nn.Module):
    """A minimal GPT-style transformer"""

    def __init__(self, config: TinyConfig):
        super().__init__()
        self.embedding = PositionalEmbedding(config)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.norm = LayerNormalization(config)
        self.proj = nn.Linear(config.d_model, config.vocab_size)  # Final projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        return self.proj(self.norm(x))  # Normalize and project to vocab size


# Usage example
if __name__ == "__main__":
    from tiny.tokenizer import TinyTokenizer

    config = TinyConfig()
    tokenizer = TinyTokenizer(config)  # Set the vocab size
    tiny = TinyTransformer(config)

    print(config)
    print(tiny)
