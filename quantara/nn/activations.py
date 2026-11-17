from __future__ import annotations

import numpy as np

from quantara.nn.module import Module, register

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class _Elementwise(Module):
    def forward(self, x):
        if self.training:
            self._x = x
        return self.f(x)

    def backward(self, grad):
        return grad * self.df(self._x)

@register
class ReLU(_Elementwise):
    f = staticmethod(lambda x: np.maximum(x, 0))
    df = staticmethod(lambda x: (x > 0).astype(x.dtype))

@register
class Tanh(_Elementwise):
    f = staticmethod(np.tanh)
    df = staticmethod(lambda x: 1.0 - np.tanh(x) ** 2)

@register
class Sigmoid(_Elementwise):
    f = staticmethod(_sigmoid)
    df = staticmethod(lambda x: _sigmoid(x) * (1.0 - _sigmoid(x)))

@register
class SiLU(_Elementwise):
    f = staticmethod(lambda x: x * _sigmoid(x))

    @staticmethod
    def df(x):
        s = _sigmoid(x)
        return s * (1.0 + x * (1.0 - s))

_GC = np.sqrt(2.0 / np.pi)

@register
class GELU(_Elementwise):
    @staticmethod
    def f(x):
        return 0.5 * x * (1.0 + np.tanh(_GC * (x + 0.044715 * x**3)))

    @staticmethod
    def df(x):
        t = np.tanh(_GC * (x + 0.044715 * x**3))
        return 0.5 * (1 + t) + 0.5 * x * (1 - t**2) * _GC * (1 + 3 * 0.044715 * x**2)
