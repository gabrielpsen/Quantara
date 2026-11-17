from __future__ import annotations

from typing import Tuple

import numpy as np

from quantara.nn.module import softmax

def cross_entropy(logits: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    p = softmax(logits, axis=-1)
    n = logits.shape[0]
    loss = -np.log(np.maximum(p[np.arange(n), y], 1e-12)).mean()
    g = p.copy()
    g[np.arange(n), y] -= 1.0
    return float(loss), g / n
