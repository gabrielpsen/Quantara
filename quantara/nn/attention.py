from __future__ import annotations

import numpy as np

from quantara.nn.layers import Linear
from quantara.nn.module import _rng, Module, register, softmax

@register
class MultiHeadAttention(Module):
    def __init__(self, dim: int, heads: int, causal: bool = True, seed=0, empty: bool = False):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim, self.heads, self.causal = int(dim), int(heads), bool(causal)
        if not empty:
            rng = _rng(seed)
            for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                setattr(self, name, Linear(dim, dim, bias=True, seed=rng))

    def init_args(self):
        return {"dim": self.dim, "heads": self.heads, "causal": self.causal, "empty": True}

    def forward(self, x):
        b, t, d = x.shape
        h, dh = self.heads, d // self.heads

        def split(a):
            return a.reshape(b, t, h, dh).transpose(0, 2, 1, 3)

        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh)
        if self.causal:
            att = np.where(np.tril(np.ones((t, t), dtype=bool)), att, -1e9)
        out = (softmax(att) @ v).transpose(0, 2, 1, 3).reshape(b, t, d)
        return self.out_proj(out)
