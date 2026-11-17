from __future__ import annotations

import numpy as np

def smooth_scales(w: np.ndarray, act_absmax: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    wmax = np.maximum(np.abs(w).max(axis=0), 1e-5)
    amax = np.maximum(act_absmax, 1e-5)
    s = (amax**alpha) / (wmax ** (1.0 - alpha))
    return np.clip(s, 1e-5, 1e5).astype(np.float32)
