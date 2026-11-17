from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import numpy as np

from quantara.core.formats import get_quantizer, QuantConfig

@dataclass(frozen=True)
class ActConfig:
    bits: int = 8
    symmetric: bool = True
    dynamic: bool = True
    observer: str = "minmax"
    percentile: float = 99.99

    def __post_init__(self):
        if not 2 <= self.bits <= 8:
            raise ValueError("activation bits must be in [2, 8]")
        if self.observer not in ("minmax", "percentile"):
            raise ValueError("activation observer must be minmax or percentile")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ActConfig:
        return cls(**d)

    def quant_config(self) -> QuantConfig:
        return QuantConfig(
            bits=self.bits,
            format="int",
            symmetric=self.symmetric,
            granularity="channel",
            scale_dtype="float32",
        )

def static_params(mn: float, mx: float, bits: int, symmetric: bool) -> Tuple[np.ndarray, np.ndarray]:
    qmax = 2**bits - 1
    if symmetric:
        scale = max(max(abs(mn), abs(mx)) / (2 ** (bits - 1) - 1), 1e-12)
        zero = float(2 ** (bits - 1))
    else:
        mn, mx = min(mn, 0.0), max(mx, 0.0)
        scale = max((mx - mn) / qmax, 1e-12)
        zero = float(np.clip(np.rint(-mn / scale), 0, qmax))
    return np.float32(scale), np.float32(zero)

def fake_quant_act(x: np.ndarray, cfg: ActConfig, scale=None, zero=None) -> np.ndarray:
    if cfg.dynamic or scale is None:
        qz = get_quantizer(cfg.quant_config())
        x2 = np.asarray(x, dtype=np.float32).reshape(-1, x.shape[-1])
        sc, z = qz.fit(x2)
        return qz.qdq(x2, sc, z).reshape(x.shape)
    qmax = 2**cfg.bits - 1
    q = np.clip(np.rint(x / scale) + zero, 0, qmax)
    return ((q - zero) * scale).astype(np.float32)

class EMAObserver:
    def __init__(self, momentum: float = 0.9):
        self.momentum = momentum
        self.mn: Optional[float] = None
        self.mx: Optional[float] = None

    def update(self, x):
        mn, mx = float(x.min()), float(x.max())
        if self.mn is None:
            self.mn, self.mx = mn, mx
        else:
            m = self.momentum
            self.mn = m * self.mn + (1 - m) * mn
            self.mx = m * self.mx + (1 - m) * mx

    def range(self):
        return (0.0, 0.0) if self.mn is None else (self.mn, self.mx)
