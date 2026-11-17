from __future__ import annotations

from typing import List, Optional

from quantara.nn.module import Module, register

@register
class Sequential(Module):
    def __init__(self, layers: Optional[List[Module]] = None):
        super().__init__()
        for i, layer in enumerate(layers or []):
            self._modules[str(i)] = layer

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x

    def backward(self, grad):
        for m in reversed(list(self._modules.values())):
            grad = m.backward(grad)
        return grad

    def __len__(self):
        return len(self._modules)

    def __getitem__(self, i):
        return list(self._modules.values())[i]

    @classmethod
    def from_spec(cls, args, children):
        return cls(list(children.values()))

@register
class Residual(Module):
    def __init__(self, body: Optional[Module] = None):
        super().__init__()
        if body is not None:
            self.body = body

    def forward(self, x):
        return x + self.body(x)

    def backward(self, grad):
        return grad + self.body.backward(grad)
