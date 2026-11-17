from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from quantara.nn.module import Module
from quantara.training.losses import cross_entropy
from quantara.training.optim import Adam, Optimizer

def fit(
    model: Module,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    optimizer: Optional[Optimizer] = None,
    loss_fn: Callable = cross_entropy,
    seed: int = 0,
    verbose: bool = False,
) -> List[float]:
    opt = optimizer or Adam(model.parameters(), lr=3e-3)
    rng = np.random.default_rng(seed)
    n = len(x)
    history = []
    model.train()
    for ep in range(epochs):
        order = rng.permutation(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = order[i : i + batch_size]
            out = model(x[idx])
            loss, g = loss_fn(out, y[idx])
            opt.zero_grad()
            model.backward(g)
            opt.step()
            total += loss * len(idx)
        history.append(total / n)
        if verbose:
            print(f"epoch {ep + 1:3d}/{epochs}  loss {history[-1]:.4f}")
    return history

def accuracy(model: Module, x: np.ndarray, y: np.ndarray, batch_size: int = 1024) -> float:
    was = model.training
    model.eval()
    correct = 0
    for i in range(0, len(x), batch_size):
        correct += int((model(x[i : i + batch_size]).argmax(axis=-1) == y[i : i + batch_size]).sum())
    model.train(was)
    return correct / len(x)
