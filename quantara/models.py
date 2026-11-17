from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from quantara.nn.activations import GELU, ReLU
from quantara.nn.attention import MultiHeadAttention
from quantara.nn.containers import Residual, Sequential
from quantara.nn.layers import Conv2d, Embedding, Flatten, LayerNorm, Linear
from quantara.nn.module import Module

def mlp(sizes: Sequence[int], activation=ReLU, seed: int = 0) -> Sequential:
    rng = np.random.default_rng(seed)
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(Linear(sizes[i], sizes[i + 1], seed=rng))
        if i < len(sizes) - 2:
            layers.append(activation())
    return Sequential(layers)

def small_cnn(in_ch: int = 1, classes: int = 4, size: int = 12, seed: int = 0) -> Sequential:
    rng = np.random.default_rng(seed)
    side = size - 2
    return Sequential(
        [
            Conv2d(in_ch, 8, 3, seed=rng),
            ReLU(),
            Conv2d(8, 8, 3, seed=rng),
            ReLU(),
            Flatten(),
            Linear(8 * (side - 2) ** 2, classes, seed=rng),
        ]
    )

def tiny_lm(vocab: int = 256, dim: int = 64, heads: int = 4, layers: int = 2, seed: int = 0) -> Sequential:
    rng = np.random.default_rng(seed)
    blocks = [Embedding(vocab, dim, seed=rng)]
    for _ in range(layers):
        blocks.append(Residual(Sequential([LayerNorm(dim), MultiHeadAttention(dim, heads, seed=rng)])))
        blocks.append(
            Residual(
                Sequential(
                    [LayerNorm(dim), Linear(dim, 4 * dim, seed=rng), GELU(), Linear(4 * dim, dim, seed=rng)]
                )
            )
        )
    blocks += [LayerNorm(dim), Linear(dim, vocab, bias=False, seed=rng)]
    return Sequential(blocks)

def inject_outliers(model: Module, fraction: float = 0.03, factor: float = 25.0, seed: int = 0) -> Module:
    rng = np.random.default_rng(seed)
    for m in model.modules():
        if isinstance(m, LayerNorm):
            k = max(1, int(m.dim * fraction))
            idx = rng.choice(m.dim, k, replace=False)
            m.weight.data[idx] *= factor
    return model

def make_spirals(
    n: int = 1500, classes: int = 3, noise: float = 0.1, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    per = n // classes
    xs, ys = [], []
    for c in range(classes):
        t = np.linspace(0.15, 1.0, per)
        ang = t * 2.2 * np.pi + c * 2 * np.pi / classes
        r = t * 2.0
        pts = np.stack([r * np.sin(ang), r * np.cos(ang)], axis=1) + rng.normal(0, noise, (per, 2))
        xs.append(pts)
        ys.append(np.full(per, c))
    x, y = np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.int64)
    p = rng.permutation(len(x))
    return x[p], y[p]

def make_patterns(n: int = 800, size: int = 12, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.25, (n, 1, size, size)).astype(np.float32)
    y = rng.integers(0, 4, n)
    for i in range(n):
        o = rng.integers(2, size - 2)
        if y[i] == 0:
            x[i, 0, o, :] += 1.0
        elif y[i] == 1:
            x[i, 0, :, o] += 1.0
        elif y[i] == 2:
            x[i, 0] += np.eye(size, dtype=np.float32)
        else:
            x[i, 0, o - 2 : o + 2, o - 2 : o + 2] += 1.0
    return x, y.astype(np.int64)
