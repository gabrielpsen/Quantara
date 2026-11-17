from __future__ import annotations

import fnmatch
from typing import Optional

import numpy as np

from quantara.core.formats import QuantConfig
from quantara.core.observers import ActConfig, EMAObserver, fake_quant_act, static_params
from quantara.core.tensor import fake_quant, quantize_tensor
from quantara.nn.layers import Conv2d, Linear
from quantara.nn.module import Module
from quantara.quantize.layers import QuantConv2d, QuantLinear

class _QATMixin:
    def _qat_setup(self, wcfg: QuantConfig, act: Optional[ActConfig]):
        self.wcfg = wcfg
        self.act = act
        self.obs = EMAObserver(0.9)
        self._wq = None

    def _quant_input(self, x):
        self._mask = None
        if self.act is None:
            return x
        if self.act.dynamic:
            return fake_quant_act(x, self.act)
        if self.training:
            self.obs.update(x)
        if self.obs.mn is None:
            return x
        mn, mx = self.obs.range()
        scale, zero = static_params(mn, mx, self.act.bits, self.act.symmetric)
        lo = -zero * scale
        hi = (2**self.act.bits - 1 - zero) * scale
        self._mask = ((x >= lo) & (x <= hi)).astype(x.dtype)
        return fake_quant_act(x, self.act, scale, zero)

    def qat_forward(self, x, parent_forward):
        latent = self.weight.data
        self._wq = fake_quant(latent, self.wcfg).astype(latent.dtype)
        self.weight.data = self._wq
        try:
            return parent_forward(self._quant_input(x))
        finally:
            self.weight.data = latent

    def qat_backward(self, grad, parent_backward):
        latent = self.weight.data
        self.weight.data = self._wq
        try:
            dx = parent_backward(grad)
        finally:
            self.weight.data = latent
        if dx is not None and self._mask is not None:
            dx = dx * self._mask
        return dx

class QATLinear(_QATMixin, Linear):
    def __init__(self, base: Linear, wcfg: QuantConfig, act: Optional[ActConfig] = None):
        Linear.__init__(self, base.in_features, base.out_features, base.has_bias)
        self.weight.data = base.weight.data.copy()
        if base.has_bias:
            self.bias.data = base.bias.data.copy()
        self._qat_setup(wcfg, act)

    def forward(self, x):
        return self.qat_forward(x, super().forward)

    def backward(self, grad):
        return self.qat_backward(grad, super().backward)

class QATConv2d(_QATMixin, Conv2d):
    def __init__(self, base: Conv2d, wcfg: QuantConfig, act: Optional[ActConfig] = None):
        Conv2d.__init__(
            self,
            base.in_channels,
            base.out_channels,
            base.kernel_size,
            base.stride,
            base.padding,
            base.has_bias,
        )
        self.weight.data = base.weight.data.copy()
        if base.has_bias:
            self.bias.data = base.bias.data.copy()
        self._qat_setup(wcfg, act)

    def forward(self, x):
        return self.qat_forward(x, super().forward)

    def backward(self, grad):
        return self.qat_backward(grad, super().backward)

def prepare_qat(model: Module, weight: QuantConfig, act: Optional[ActConfig] = None, skip=()) -> Module:
    for path, m in list(model.named_modules()):
        if not path or any(fnmatch.fnmatch(path, p) for p in skip):
            continue
        if type(m) is Linear:
            model.set_submodule(path, QATLinear(m, weight, act))
        elif type(m) is Conv2d:
            model.set_submodule(path, QATConv2d(m, weight, act))
    return model

def convert_qat(model: Module, cache_weights: bool = True) -> Module:
    model.eval()
    for path, m in list(model.named_modules()):
        if not path or not isinstance(m, _QATMixin):
            continue
        qt = quantize_tensor(m.weight.data, m.wcfg)
        bias = m.bias.data.astype(np.float32) if m.has_bias else None
        params = None
        if m.act is not None and not m.act.dynamic:
            params = static_params(*m.obs.range(), m.act.bits, m.act.symmetric)
        if isinstance(m, Linear):
            new = QuantLinear.from_float(qt, bias, None, m.act, params, cache_weights)
        else:
            new = QuantConv2d.from_float(qt, bias, m.stride, m.padding, m.act, params, cache_weights)
        model.set_submodule(path, new)
    return model
