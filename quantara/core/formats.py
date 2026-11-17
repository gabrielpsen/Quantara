from __future__ import annotations

import functools
from dataclasses import asdict, dataclass, replace
from typing import Optional, Tuple

import numpy as np

SCALE_FLOOR = 2.0**-24

FORMATS = ("int", "nf4", "fp4", "fp8_e4m3", "fp8_e5m2", "binary", "ternary")

GRANULARITIES = ("tensor", "channel", "group")

OBSERVERS = ("minmax", "percentile", "mse")

_FIXED_BITS = {"nf4": 4, "fp4": 4, "fp8_e4m3": 8, "fp8_e5m2": 8, "binary": 1, "ternary": 2}

@dataclass(frozen=True)
class QuantConfig:
    bits: int = 4
    format: str = "int"
    symmetric: bool = True
    granularity: str = "group"
    group_size: int = 128
    axis: int = 0
    observer: str = "minmax"
    percentile: float = 99.9
    scale_dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.format not in FORMATS:
            raise ValueError(f"unknown format {self.format!r}; choose from {FORMATS}")
        if self.granularity not in GRANULARITIES:
            raise ValueError(f"unknown granularity {self.granularity!r}")
        if self.observer not in OBSERVERS:
            raise ValueError(f"unknown observer {self.observer!r}")
        if self.scale_dtype not in ("float16", "float32"):
            raise ValueError("scale_dtype must be float16 or float32")
        if self.format in _FIXED_BITS:
            object.__setattr__(self, "bits", _FIXED_BITS[self.format])
        elif not 2 <= int(self.bits) <= 8:
            raise ValueError("integer bits must be in [2, 8]")
        if self.granularity == "group" and self.group_size < 1:
            raise ValueError("group_size must be >= 1")
        if not 50.0 < self.percentile <= 100.0:
            raise ValueError("percentile must be in (50, 100]")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> QuantConfig:
        return cls(**d)

    def with_(self, **kw) -> QuantConfig:
        return replace(self, **kw)

    @property
    def name(self) -> str:
        base = f"{self.format}{self.bits}" if self.format == "int" else self.format
        sym = "" if self.format != "int" else ("-sym" if self.symmetric else "-asym")
        gran = {"tensor": "-tensor", "channel": "-channel", "group": f"-g{self.group_size}"}[self.granularity]
        return base + sym + gran

def _finalize_scale(scale: np.ndarray, dtype: str) -> np.ndarray:
    scale = np.maximum(np.asarray(scale, dtype=np.float32), np.float32(SCALE_FLOOR))
    if dtype == "float16":
        # round to fp16 before choosing codes so the stored scale is the one that was used.
        scale = scale.astype(np.float16).astype(np.float32)
        scale = np.maximum(scale, np.float32(SCALE_FLOOR))
    return scale

class Quantizer:
    bits: int

    def fit(
        self, g: np.ndarray, clip: float = 1.0, pct: Optional[float] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        raise NotImplementedError

    def encode(self, g, scale, zero) -> np.ndarray:
        raise NotImplementedError

    def decode(self, codes, scale, zero) -> np.ndarray:
        raise NotImplementedError

    def qdq(self, g, scale, zero) -> np.ndarray:
        return self.decode(self.encode(g, scale, zero), scale, zero)

class IntQuantizer(Quantizer):
    def __init__(self, bits: int, symmetric: bool, scale_dtype: str = "float16"):
        self.bits = bits
        self.symmetric = symmetric
        self.scale_dtype = scale_dtype
        self.qmax = float(2**bits - 1)
        self.qpos = float(2 ** (bits - 1) - 1)
        self.zp = float(2 ** (bits - 1))

    def fit(self, g, clip=1.0, pct=None):
        g = np.asarray(g, dtype=np.float32)
        if pct is None:
            mn = g.min(axis=1, keepdims=True)
            mx = g.max(axis=1, keepdims=True)
        else:
            mn = np.percentile(g, 100.0 - pct, axis=1, keepdims=True).astype(np.float32)
            mx = np.percentile(g, pct, axis=1, keepdims=True).astype(np.float32)
        if self.symmetric:
            amax = np.maximum(np.abs(mn), np.abs(mx)) * clip
            return _finalize_scale(amax / self.qpos, self.scale_dtype), None
        mn = np.minimum(mn, 0.0) * clip
        mx = np.maximum(mx, 0.0) * clip
        scale = _finalize_scale((mx - mn) / self.qmax, self.scale_dtype)
        zero = np.clip(np.rint(-mn / scale), 0, self.qmax).astype(np.float32)
        return scale, zero

    def encode(self, g, scale, zero):
        z = self.zp if zero is None else zero
        q = np.clip(np.rint(g / scale) + z, 0, self.qmax)
        return q.astype(np.uint8)

    def decode(self, codes, scale, zero):
        z = self.zp if zero is None else zero
        return (codes.astype(np.float32) - z) * scale

class CodebookQuantizer(Quantizer):
    def __init__(self, values, bits: int, scale_dtype: str = "float16", mode: str = "absmax"):
        self.values = np.asarray(values, dtype=np.float32)
        self.bits = bits
        self.scale_dtype = scale_dtype
        self.mode = mode
        self.vmax = float(np.abs(self.values).max())
        self.order = np.argsort(self.values, kind="stable")
        sv = self.values[self.order]
        self.mids = ((sv[1:] + sv[:-1]) / 2.0).astype(np.float32)

    def fit(self, g, clip=1.0, pct=None):
        g = np.asarray(g, dtype=np.float32)
        a = np.abs(g)
        if self.mode == "mean_abs":
            scale = a.mean(axis=1, keepdims=True) * clip
        elif self.mode == "twn":
            delta = 0.7 * a.mean(axis=1, keepdims=True)
            mask = a > delta
            cnt = np.maximum(mask.sum(axis=1, keepdims=True), 1)
            scale = (a * mask).sum(axis=1, keepdims=True) / cnt * clip
        else:
            amax = (
                a.max(axis=1, keepdims=True)
                if pct is None
                else np.percentile(a, pct, axis=1, keepdims=True).astype(np.float32)
            )
            scale = amax * clip / self.vmax
        return _finalize_scale(scale, self.scale_dtype), None

    def encode(self, g, scale, zero):
        idx = np.searchsorted(self.mids, g / scale)
        return self.order[idx].astype(np.uint8)

    def decode(self, codes, scale, zero):
        return self.values[codes] * scale

_NF4 = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]

_FP4 = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

def _fp8_table(ebits: int, mbits: int, bias: int, kind: str) -> np.ndarray:
    vals = np.zeros(256, dtype=np.float64)
    valid = np.ones(256, dtype=bool)
    for code in range(256):
        s = -1.0 if code >> 7 else 1.0
        e = (code >> mbits) & ((1 << ebits) - 1)
        m = code & ((1 << mbits) - 1)
        if e == 0:
            v = s * (m / (1 << mbits)) * 2.0 ** (1 - bias)
        else:
            v = s * (1 + m / (1 << mbits)) * 2.0 ** (e - bias)
        if kind == "e4m3" and e == 15 and m == 7:
            valid[code] = False
        if kind == "e5m2" and e == 31:
            valid[code] = False
        vals[code] = v
    vmax = np.abs(vals[valid]).max()
    for code in np.where(~valid)[0]:
        vals[code] = vmax * (1.0 if code < 128 else -1.0)
    return vals.astype(np.float32)

@functools.cache
def _cached_quantizer(fmt: str, bits: int, symmetric: bool, scale_dtype: str) -> Quantizer:
    if fmt == "int":
        return IntQuantizer(bits, symmetric, scale_dtype)
    if fmt == "nf4":
        return CodebookQuantizer(_NF4, 4, scale_dtype)
    if fmt == "fp4":
        return CodebookQuantizer(_FP4, 4, scale_dtype)
    if fmt == "fp8_e4m3":
        return CodebookQuantizer(_fp8_table(4, 3, 7, "e4m3"), 8, scale_dtype)
    if fmt == "fp8_e5m2":
        return CodebookQuantizer(_fp8_table(5, 2, 15, "e5m2"), 8, scale_dtype)
    if fmt == "binary":
        return CodebookQuantizer([-1.0, 1.0], 1, scale_dtype, mode="mean_abs")
    if fmt == "ternary":
        return CodebookQuantizer([-1.0, 0.0, 1.0], 2, scale_dtype, mode="twn")
    raise ValueError(fmt)

def get_quantizer(cfg: QuantConfig) -> Quantizer:
    return _cached_quantizer(cfg.format, cfg.bits, cfg.symmetric, cfg.scale_dtype)

MSE_RATIOS = np.linspace(1.0, 0.5, 26)

def fit_params(qz: Quantizer, g: np.ndarray, cfg: QuantConfig):
    g = np.asarray(g, dtype=np.float32)
    if cfg.observer == "minmax" or cfg.format in ("binary", "ternary"):
        return qz.fit(g)
    if cfg.observer == "percentile":
        return qz.fit(g, pct=cfg.percentile)
    best_err = None
    best_scale = best_zero = None
    for r in MSE_RATIOS:
        sc, z = qz.fit(g, clip=float(r))
        err = ((qz.qdq(g, sc, z) - g) ** 2).sum(axis=1, keepdims=True)
        if best_err is None:
            best_err, best_scale, best_zero = err, sc, z
        else:
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_scale = np.where(better, sc, best_scale)
            if z is not None:
                best_zero = np.where(better, z, best_zero)
    return best_scale, best_zero
