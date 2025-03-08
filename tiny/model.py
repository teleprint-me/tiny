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
