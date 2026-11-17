from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from quantara.nn.module import _rng, Module, Param, register

@register
class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, seed=0):
        super().__init__()
        self.in_features, self.out_features = int(in_features), int(out_features)
        rng = _rng(seed)
        # cast last: a float64 scalar would promote the array under numpy 2.
        w = (rng.standard_normal((out_features, in_features)) * np.sqrt(2.0 / in_features)).astype(np.float32)
        self.weight = Param(w)
        self.has_bias = bool(bias)
        if bias:
            self.bias = Param(np.zeros(out_features, dtype=np.float32))
        self._x = None

    @classmethod
    def from_arrays(cls, weight: np.ndarray, bias: Optional[np.ndarray] = None) -> Linear:
        obj = cls.__new__(cls)
        Module.__init__(obj)
        obj.out_features, obj.in_features = (int(s) for s in weight.shape)
        obj.weight = Param(weight)
        obj.has_bias = bias is not None
        if bias is not None:
            obj.bias = Param(bias)
        obj._x = None
        return obj

    def init_args(self):
        return {"in_features": self.in_features, "out_features": self.out_features, "bias": self.has_bias}

    @classmethod
    def from_spec(cls, args, children):
        obj = cls.from_arrays(np.empty((0, 0), np.float32), np.empty(0, np.float32) if args["bias"] else None)
        obj.in_features, obj.out_features = args["in_features"], args["out_features"]
        return obj

    def forward(self, x):
        if self.training:
            self._x = x
        y = x @ self.weight.data.T
        if self.has_bias:
            y = y + self.bias.data
        return y

    def backward(self, grad):
        if self._x is None:
            raise RuntimeError("backward() called without a training-mode forward()")
        x2 = self._x.reshape(-1, self.in_features)
        g2 = grad.reshape(-1, self.out_features)
        self.weight.accumulate(g2.T @ x2)
        if self.has_bias:
            self.bias.accumulate(g2.sum(axis=0))
        return grad @ self.weight.data

    def extra_repr(self):
        return f"{self.in_features}, {self.out_features}, bias={self.has_bias}"

def im2col(x: np.ndarray, kh: int, kw: int, stride: int, pad: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    win = sliding_window_view(x, (kh, kw), axis=(2, 3))[:, :, ::stride, ::stride]
    n, c, oh, ow = win.shape[:4]
    cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(n * oh * ow, c * kh * kw)
    return cols, (oh, ow)

def conv2d_forward(x, w, b, stride, pad):
    o = w.shape[0]
    kh, kw = w.shape[2:]
    cols, (oh, ow) = im2col(x, kh, kw, stride, pad)
    y = cols @ w.reshape(o, -1).T
    if b is not None:
        y = y + b
    return y.reshape(x.shape[0], oh, ow, o).transpose(0, 3, 1, 2), cols

@register
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, seed=0):
        super().__init__()
        self.in_channels, self.out_channels = int(in_channels), int(out_channels)
        self.kernel_size, self.stride, self.padding = int(kernel_size), int(stride), int(padding)
        rng = _rng(seed)
        fan_in = in_channels * kernel_size * kernel_size
        w = rng.standard_normal((out_channels, in_channels, kernel_size, kernel_size))
        self.weight = Param((w * np.sqrt(2.0 / fan_in)).astype(np.float32))
        self.has_bias = bool(bias)
        if bias:
            self.bias = Param(np.zeros(out_channels, dtype=np.float32))
        self._cache = None

    def init_args(self):
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "bias": self.has_bias,
        }

    def forward(self, x):
        b = self.bias.data if self.has_bias else None
        y, cols = conv2d_forward(x, self.weight.data, b, self.stride, self.padding)
        if self.training:
            self._cache = (x.shape, cols)
        return y

    def backward(self, grad):
        if self._cache is None:
            raise RuntimeError("backward() called without a training-mode forward()")
        xshape, cols = self._cache
        n, c, h, w = xshape
        k, s, p = self.kernel_size, self.stride, self.padding
        o = self.out_channels
        g2 = grad.transpose(0, 2, 3, 1).reshape(-1, o)
        self.weight.accumulate((g2.T @ cols).reshape(self.weight.data.shape))
        if self.has_bias:
            self.bias.accumulate(g2.sum(axis=0))
        gcols = g2 @ self.weight.data.reshape(o, -1)
        oh, ow = grad.shape[2:]
        gcols = gcols.reshape(n, oh, ow, c, k, k)
        dxp = np.zeros((n, c, h + 2 * p, w + 2 * p), dtype=grad.dtype)
        for i in range(k):
            for j in range(k):
                dxp[:, :, i : i + s * oh : s, j : j + s * ow : s] += gcols[:, :, :, :, i, j].transpose(
                    0, 3, 1, 2
                )
        return dxp[:, :, p : p + h, p : p + w] if p else dxp

    @classmethod
    def from_spec(cls, args, children):
        obj = cls.__new__(cls)
        Module.__init__(obj)
        obj.in_channels, obj.out_channels = args["in_channels"], args["out_channels"]
        obj.kernel_size, obj.stride, obj.padding = args["kernel_size"], args["stride"], args["padding"]
        obj.weight = Param(np.empty(0, np.float32))
        obj.has_bias = args["bias"]
        if obj.has_bias:
            obj.bias = Param(np.empty(0, np.float32))
        obj._cache = None
        return obj

    def extra_repr(self):
        k, s, p = self.kernel_size, self.stride, self.padding
        return f"{self.in_channels}, {self.out_channels}, k={k}, s={s}, p={p}"

@register
class Flatten(Module):
    def forward(self, x):
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self._shape)

@register
class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim, self.eps = int(dim), float(eps)
        self.weight = Param(np.ones(dim, dtype=np.float32))
        self.bias = Param(np.zeros(dim, dtype=np.float32))

    def init_args(self):
        return {"dim": self.dim, "eps": self.eps}

    @classmethod
    def from_spec(cls, args, children):
        obj = cls.__new__(cls)
        Module.__init__(obj)
        obj.dim, obj.eps = args["dim"], args["eps"]
        obj.weight = Param(np.empty(0, np.float32))
        obj.bias = Param(np.empty(0, np.float32))
        return obj

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(var + self.eps)
        xhat = (x - mu) * inv
        if self.training:
            self._cache = (xhat, inv)
        return xhat * self.weight.data + self.bias.data

    def backward(self, grad):
        xhat, inv = self._cache
        flat = (-1, self.dim)
        self.weight.accumulate((grad * xhat).reshape(flat).sum(axis=0))
        self.bias.accumulate(grad.reshape(flat).sum(axis=0))
        dxhat = grad * self.weight.data
        return inv * (
            dxhat - dxhat.mean(axis=-1, keepdims=True) - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)
        )

@register
class Embedding(Module):
    def __init__(self, num_embeddings: int, dim: int, seed=0):
        super().__init__()
        self.num_embeddings, self.dim = int(num_embeddings), int(dim)
        self.weight = Param(_rng(seed).standard_normal((num_embeddings, dim)).astype(np.float32) * 0.5)
        self._idx = None

    def init_args(self):
        return {"num_embeddings": self.num_embeddings, "dim": self.dim}

    @classmethod
    def from_spec(cls, args, children):
        obj = cls.__new__(cls)
        Module.__init__(obj)
        obj.num_embeddings, obj.dim = args["num_embeddings"], args["dim"]
        obj.weight = Param(np.empty(0, np.float32))
        obj._idx = None
        return obj

    def forward(self, idx):
        idx = np.asarray(idx)
        if self.training:
            self._idx = idx
        return self.weight.data[idx]

    def backward(self, grad):
        g = np.zeros_like(self.weight.data)
        np.add.at(g, self._idx, grad)
        self.weight.accumulate(g)
