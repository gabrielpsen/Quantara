from __future__ import annotations

from typing import Optional

import numpy as np

from quantara.core.observers import ActConfig, fake_quant_act
from quantara.core.tensor import QTensor
from quantara.nn.layers import conv2d_forward
from quantara.nn.module import Module, register

class _QuantBase(Module):
    def __init__(self, qmeta: dict, act: Optional[dict], cache_weights: bool):
        super().__init__()
        self._qmeta = dict(qmeta)
        self._wcache: Optional[np.ndarray] = None
        self.act = None if act is None else ActConfig.from_dict(act)
        self.cache_weights = bool(cache_weights)

    def _set_qweight(self, qt: QTensor) -> None:
        self.set_buffer("packed", qt.packed)
        self.set_buffer("scale", qt.scale)
        self.set_buffer("zero", qt.zero)
        self._qmeta = qt.meta()
        self._wcache = None

    def _set_act_params(self, act_params) -> None:
        if act_params is not None:
            self.set_buffer("act_scale", np.asarray(act_params[0], dtype=np.float32))
            self.set_buffer("act_zero", np.asarray(act_params[1], dtype=np.float32))

    @property
    def qweight(self) -> QTensor:
        arrays = {"packed": self._buffers["packed"], "scale": self._buffers["scale"]}
        if "zero" in self._buffers:
            arrays["zero"] = self._buffers["zero"]
        return QTensor.from_parts(arrays, self._qmeta)

    def dequantized_weight(self) -> np.ndarray:
        if not self.cache_weights:
            return self.qweight.dequantize()
        if self._wcache is None:
            self._wcache = self.qweight.dequantize()
        return self._wcache

    def _prep_input(self, x):
        if "in_scale" in self._buffers:
            x = x * self._buffers["in_scale"]
        if self.act is not None:
            x = fake_quant_act(x, self.act, self._buffers.get("act_scale"), self._buffers.get("act_zero"))
        return x

    def _common_args(self):
        return {
            "qmeta": self._qmeta,
            "act": None if self.act is None else self.act.to_dict(),
            "cache_weights": self.cache_weights,
        }

    def own_size_bytes(self) -> int:
        n = self.qweight.nbytes
        for k in ("bias", "in_scale", "act_scale", "act_zero"):
            if k in self._buffers:
                n += self._buffers[k].nbytes
        return n

@register
class QuantLinear(_QuantBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        qmeta: dict,
        act: Optional[dict] = None,
        cache_weights: bool = True,
    ):
        super().__init__(qmeta, act, cache_weights)
        self.in_features, self.out_features = int(in_features), int(out_features)

    @classmethod
    def from_float(
        cls,
        qt: QTensor,
        bias: Optional[np.ndarray],
        in_scale=None,
        act: Optional[ActConfig] = None,
        act_params=None,
        cache_weights: bool = True,
    ) -> QuantLinear:
        out_f, in_f = qt.shape
        m = cls(in_f, out_f, qt.meta(), None if act is None else act.to_dict(), cache_weights)
        m._set_qweight(qt)
        m.set_buffer("bias", bias)
        m.set_buffer("in_scale", in_scale)
        m._set_act_params(act_params)
        return m

    def init_args(self):
        return {"in_features": self.in_features, "out_features": self.out_features, **self._common_args()}

    def forward(self, x):
        x = self._prep_input(np.asarray(x, dtype=np.float32))
        y = x @ self.dequantized_weight().T
        if "bias" in self._buffers:
            y = y + self._buffers["bias"]
        return y

    def extra_repr(self):
        a = "" if self.act is None else f", act=int{self.act.bits}{'-dyn' if self.act.dynamic else '-static'}"
        return f"{self.in_features}, {self.out_features}, w={self.qweight.cfg.name}{a}"

@register
class QuantConv2d(_QuantBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        qmeta: dict,
        act: Optional[dict] = None,
        cache_weights: bool = True,
    ):
        super().__init__(qmeta, act, cache_weights)
        self.in_channels, self.out_channels = int(in_channels), int(out_channels)
        self.kernel_size, self.stride, self.padding = int(kernel_size), int(stride), int(padding)

    @classmethod
    def from_float(
        cls,
        qt: QTensor,
        bias,
        stride,
        padding,
        act: Optional[ActConfig] = None,
        act_params=None,
        cache_weights: bool = True,
    ) -> QuantConv2d:
        o, c, k, _ = qt.shape
        m = cls(c, o, k, stride, padding, qt.meta(), None if act is None else act.to_dict(), cache_weights)
        m._set_qweight(qt)
        m.set_buffer("bias", bias)
        m._set_act_params(act_params)
        return m

    def init_args(self):
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            **self._common_args(),
        }

    def forward(self, x):
        x = self._prep_input(np.asarray(x, dtype=np.float32))
        y, _ = conv2d_forward(
            x, self.dequantized_weight(), self._buffers.get("bias"), self.stride, self.padding
        )
        return y

    def extra_repr(self):
        return f"{self.in_channels}, {self.out_channels}, k={self.kernel_size}, w={self.qweight.cfg.name}"

@register
class QuantEmbedding(_QuantBase):
    def __init__(self, num_embeddings: int, dim: int, qmeta: dict, cache_weights: bool = False):
        super().__init__(qmeta, None, cache_weights)
        self.num_embeddings, self.dim = int(num_embeddings), int(dim)

    @classmethod
    def from_float(cls, qt: QTensor, cache_weights: bool = False) -> QuantEmbedding:
        n, d = qt.shape
        m = cls(n, d, qt.meta(), cache_weights)
        m._set_qweight(qt)
        return m

    def init_args(self):
        return {
            "num_embeddings": self.num_embeddings,
            "dim": self.dim,
            "qmeta": self._qmeta,
            "cache_weights": self.cache_weights,
        }

    def forward(self, idx):
        idx = np.asarray(idx)
        if self.cache_weights:
            return self.dequantized_weight()[idx]
        return self.qweight.dequantize_rows(idx)

    def extra_repr(self):
        return f"{self.num_embeddings}, {self.dim}, w={self.qweight.cfg.name}"
