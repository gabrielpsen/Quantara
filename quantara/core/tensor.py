from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from quantara.core.formats import fit_params, get_quantizer, QuantConfig
from quantara.core.packing import pack, packed_nbytes, unpack

def _matrixify(x: np.ndarray, axis: int) -> np.ndarray:
    if x.ndim == 1:
        return x.reshape(1, -1)
    return np.moveaxis(x, axis, 0).reshape(x.shape[axis], -1)

def _unmatrixify(m: np.ndarray, shape: Tuple[int, ...], axis: int) -> np.ndarray:
    if len(shape) == 1:
        return m.reshape(shape)
    moved = (shape[axis],) + tuple(s for i, s in enumerate(shape) if i != axis)
    return np.moveaxis(m.reshape(moved), 0, axis)

def to_groups(x: np.ndarray, cfg: QuantConfig) -> Tuple[np.ndarray, int]:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 0:
        x = x.reshape(1)
    if cfg.granularity == "tensor":
        return x.reshape(1, -1), 0
    axis = cfg.axis % x.ndim if x.ndim > 1 else 0
    m = _matrixify(x, axis)
    if cfg.granularity == "channel":
        return np.ascontiguousarray(m), 0
    length = min(cfg.group_size, m.shape[1])
    pad = (-m.shape[1]) % length
    if pad:
        m = np.pad(m, ((0, 0), (0, pad)))
    return np.ascontiguousarray(m.reshape(-1, length)), pad

def from_groups(g: np.ndarray, shape: Tuple[int, ...], cfg: QuantConfig, pad: int) -> np.ndarray:
    if cfg.granularity == "tensor":
        return g.reshape(shape)
    ndim = len(shape)
    axis = cfg.axis % ndim if ndim > 1 else 0
    channels = shape[axis] if ndim > 1 else 1
    m = g.reshape(channels, -1)
    if pad:
        m = m[:, :-pad]
    return _unmatrixify(m, shape, axis)

@dataclass
class QTensor:
    packed: np.ndarray
    scale: np.ndarray
    zero: Optional[np.ndarray]
    cfg: QuantConfig
    shape: Tuple[int, ...]
    group_shape: Tuple[int, int]
    pad: int = 0

    @classmethod
    def create(cls, codes, scale, zero, cfg: QuantConfig, shape, pad: int = 0) -> QTensor:
        codes = np.asarray(codes, dtype=np.uint8)
        z = None if zero is None else np.asarray(zero).astype(np.uint8)
        return cls(
            pack(codes, cfg.bits),
            np.asarray(scale, np.float32),
            z,
            cfg,
            tuple(int(s) for s in shape),
            (int(codes.shape[0]), int(codes.shape[1])),
            int(pad),
        )

    @property
    def codes(self) -> np.ndarray:
        n, length = self.group_shape
        return unpack(self.packed, self.cfg.bits, n * length).reshape(n, length)

    def dequantize(self, dtype=np.float32) -> np.ndarray:
        qz = get_quantizer(self.cfg)
        zero = None if self.zero is None else self.zero.astype(np.float32)
        g = qz.decode(self.codes, self.scale, zero)
        return from_groups(g, self.shape, self.cfg, self.pad).astype(dtype, copy=False)

    def dequantize_rows(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows)
        if self.cfg.granularity == "tensor" or len(self.shape) != 2 or self.cfg.axis % 2 != 0:
            return self.dequantize()[rows]
        n, length = self.group_shape
        r, cols = self.shape
        ng = n // r
        flat = rows.reshape(-1)
        qz = get_quantizer(self.cfg)
        row_bits = ng * length * self.cfg.bits
        # byte-aligned rows can be sliced out without unpacking the whole table.
        if row_bits % 8 == 0:
            sub = self.packed[: r * row_bits // 8].reshape(r, row_bits // 8)[flat]
            codes = unpack(sub.reshape(-1), self.cfg.bits, flat.size * ng * length)
        else:
            codes = self.codes.reshape(r, ng, length)[flat]
        codes = codes.reshape(flat.size, ng, length)
        sc = self.scale.reshape(r, ng, 1)[flat]
        zr = None if self.zero is None else self.zero.astype(np.float32).reshape(r, ng, 1)[flat]
        out = qz.decode(codes, sc, zr).reshape(flat.size, ng * length)[:, :cols]
        return out.reshape(*rows.shape, cols)

    @property
    def nbytes(self) -> int:
        n = self.group_shape[0]
        sbytes = 2 if self.cfg.scale_dtype == "float16" else 4
        total = packed_nbytes(self.group_shape[0] * self.group_shape[1], self.cfg.bits) + n * sbytes
        if self.zero is not None:
            total += packed_nbytes(n, self.cfg.bits)
        return total

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape))

    @property
    def bits_per_weight(self) -> float:
        return 8.0 * self.nbytes / max(self.numel, 1)

    def arrays(self) -> Dict[str, np.ndarray]:
        out = {"packed": self.packed, "scale": self.scale}
        if self.zero is not None:
            out["zero"] = self.zero
        return out

    def meta(self) -> dict:
        return {
            "cfg": self.cfg.to_dict(),
            "shape": list(self.shape),
            "group_shape": list(self.group_shape),
            "pad": self.pad,
        }

    @classmethod
    def from_parts(cls, arrays: Dict[str, np.ndarray], meta: dict) -> QTensor:
        return cls(
            np.asarray(arrays["packed"], np.uint8),
            np.asarray(arrays["scale"], np.float32),
            arrays.get("zero"),
            QuantConfig.from_dict(meta["cfg"]),
            tuple(meta["shape"]),
            tuple(meta["group_shape"]),
            int(meta["pad"]),
        )

    def __repr__(self) -> str:
        return f"QTensor(shape={self.shape}, {self.cfg.name}, {self.bits_per_weight:.2f} bits/weight)"

def quantize_tensor(x: np.ndarray, cfg: QuantConfig, params=None) -> QTensor:
    x = np.asarray(x, dtype=np.float32)
    g, pad = to_groups(x, cfg)
    qz = get_quantizer(cfg)
    scale, zero = params if params is not None else fit_params(qz, g, cfg)
    codes = qz.encode(g, scale, zero)
    return QTensor.create(codes, scale, zero, cfg, x.shape if x.ndim else (1,), pad)

def fake_quant(x: np.ndarray, cfg: QuantConfig) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    g, pad = to_groups(x, cfg)
    qz = get_quantizer(cfg)
    scale, zero = fit_params(qz, g, cfg)
    return from_groups(qz.qdq(g, scale, zero), x.shape, cfg, pad)
