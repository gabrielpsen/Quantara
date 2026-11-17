from __future__ import annotations

from typing import Tuple

import numpy as np

from quantara.core.formats import QuantConfig
from quantara.core.tensor import fake_quant, QTensor, quantize_tensor

def awq(w: np.ndarray, samples: np.ndarray, cfg: QuantConfig, grid: int = 20) -> Tuple[QTensor, np.ndarray]:
    W = np.asarray(w, dtype=np.float32)
    X = np.asarray(samples, dtype=np.float32)
    x_mean = np.maximum(np.abs(X).mean(axis=0), 1e-8)
    ref = X @ W.T
    best_err, best_s = np.inf, np.ones(W.shape[1], dtype=np.float32)
    for k in range(grid + 1):
        alpha = k / grid
        s = x_mean**alpha
        s = (s / np.sqrt(s.max() * s.min())).astype(np.float32)
        s = np.maximum(s, 1e-4)
        wq = fake_quant(W * s[None, :], cfg) / s[None, :]
        err = float(np.mean((X @ wq.T - ref) ** 2))
        if err < best_err:
            best_err, best_s = err, s
    return quantize_tensor(W * best_s[None, :], cfg), (1.0 / best_s).astype(np.float32)
