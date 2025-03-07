"""
Module: tiny.model
Description: A super simple decoder-only transformer implementation for natural language processing.

---

Do **not** use dropout. This is known to harm the model.
If we use a 0.1 dropout, that means 10% of samples are randomly dropped from the propogated sequence. This can cause stuttering and repetition issues. I omit dropout from the implementation as a result.
"""

import torch
from torch import nn
