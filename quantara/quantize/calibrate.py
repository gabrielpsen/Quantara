from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

from quantara.nn.layers import Conv2d, im2col
from quantara.nn.module import Module

HESSIAN_F64_LIMIT = 8192

class LayerStats:
    def __init__(self, features: int, hessian: bool = False, max_samples: int = 2048, seed: int = 0):
        self.features = features
        self.n = 0
        self.chmin = np.full(features, np.inf, dtype=np.float32)
        self.chmax = np.full(features, -np.inf, dtype=np.float32)
        self.hessian_dtype = np.float64 if features <= HESSIAN_F64_LIMIT else np.float32
        self.hessian: Optional[np.ndarray] = (
            np.zeros((features, features), self.hessian_dtype) if hessian else None
        )
        self.max_samples = max_samples
        self._buf = np.empty((max_samples, features), dtype=np.float32) if max_samples else None
        self._seen = 0
        self._rng = np.random.default_rng(seed)

    def update(self, x2d: np.ndarray) -> None:
        x = np.asarray(x2d, dtype=np.float32)
        m = x.shape[0]
        if m == 0:
            return
        self.n += m
        self.chmin = np.minimum(self.chmin, x.min(axis=0))
        self.chmax = np.maximum(self.chmax, x.max(axis=0))
        if self.hessian is not None:
            xh = x.astype(self.hessian_dtype, copy=False)
            self.hessian += xh.T @ xh
        if self._buf is not None:
            t = self._seen + np.arange(m)
            # algorithm R, vectorised per batch.
            slot = np.where(t < self.max_samples, t, self._rng.integers(0, t + 1))
            keep = slot < self.max_samples
            self._buf[slot[keep]] = x[keep]
            self._seen += m

    @property
    def absmax(self) -> np.ndarray:
        return np.maximum(np.abs(self.chmin), np.abs(self.chmax))

    @property
    def samples(self) -> Optional[np.ndarray]:
        if self._buf is None or self._seen == 0:
            return None
        return self._buf[: min(self._seen, self.max_samples)]

    @property
    def H(self) -> Optional[np.ndarray]:
        if self.hessian is None:
            return None
        return self.hessian / self.hessian.dtype.type(max(self.n, 1))

def layer_inputs_2d(module: Module, x: np.ndarray) -> np.ndarray:
    if isinstance(module, Conv2d) or (hasattr(module, "kernel_size") and hasattr(module, "padding")):
        cols, _ = im2col(np.asarray(x), module.kernel_size, module.kernel_size, module.stride, module.padding)
        return cols
    return np.asarray(x).reshape(-1, x.shape[-1])

def features_of(module: Module) -> int:
    if hasattr(module, "in_features"):
        return module.in_features
    return module.in_channels * module.kernel_size**2

def collect_stats(
    model: Module,
    batches: Iterable,
    targets: Dict[str, Module],
    hessian: bool = False,
    max_samples: int = 2048,
) -> Dict[str, LayerStats]:
    stats: Dict[str, LayerStats] = {}
    removers = []
    was_training = {id(m): m.training for m in model.modules()}
    model.eval()

    # layers fed by the same array (q/k/v, gate/up) share one set of stats.
    shared: Dict[int, tuple] = {}

    def make_hook(path):
        def hook(module, x, out):
            st = stats.get(path)
            entry = shared.get(id(x))
            if st is None:
                st = entry[1] if entry is not None else LayerStats(features_of(module), hessian, max_samples)
                stats[path] = st
            if entry is None:
                shared[id(x)] = (x, st)
                st.update(layer_inputs_2d(module, x))

        return hook

    try:
        for path, mod in targets.items():
            removers.append(mod.register_forward_hook(make_hook(path)))
        for batch in batches:
            shared.clear()
            model(batch)
        shared.clear()
    finally:
        for r in removers:
            r()
        for m in model.modules():
            object.__setattr__(m, "training", was_training.get(id(m), True))
    return stats
