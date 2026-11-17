from __future__ import annotations

from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

REGISTRY: Dict[str, type] = {}

def register(cls):
    REGISTRY[cls.__name__] = cls
    return cls

class Param:
    def __init__(self, data: np.ndarray):
        self.data = np.asarray(data)
        self.grad: Optional[np.ndarray] = None

    def zero_grad(self) -> None:
        self.grad = None

    def accumulate(self, g: np.ndarray) -> None:
        self.grad = g.astype(self.data.dtype, copy=False) if self.grad is None else self.grad + g

class Module:
    def __init__(self) -> None:
        object.__setattr__(self, "_params", {})
        object.__setattr__(self, "_modules", {})
        object.__setattr__(self, "_buffers", {})
        object.__setattr__(self, "_hooks", [])
        object.__setattr__(self, "training", True)

    def __setattr__(self, name, value):
        if isinstance(value, Param):
            self._params[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        for store in ("_params", "_modules", "_buffers"):
            d = self.__dict__.get(store)
            if d is not None and name in d:
                return d[name]
        raise AttributeError(f"{type(self).__name__!r} has no attribute {name!r}")

    def set_buffer(self, name: str, value: Optional[np.ndarray]) -> None:
        if value is None:
            self._buffers.pop(name, None)
        else:
            self._buffers[name] = np.asarray(value)

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError(f"{type(self).__name__} does not support backward()")

    def __call__(self, x):
        out = self.forward(x)
        for hook in list(self._hooks):
            hook(self, x, out)
        return out

    def register_forward_hook(self, fn: Callable) -> Callable[[], None]:
        self._hooks.append(fn)
        return lambda: self._hooks.remove(fn) if fn in self._hooks else None

    def named_modules(self, prefix: str = "") -> Iterator[Tuple[str, Module]]:
        yield prefix, self
        for name, m in self._modules.items():
            yield from m.named_modules(f"{prefix}.{name}" if prefix else name)

    def modules(self) -> Iterator[Module]:
        for _, m in self.named_modules():
            yield m

    def named_parameters(self) -> Iterator[Tuple[str, Param]]:
        for path, m in self.named_modules():
            for n, p in m._params.items():
                yield (f"{path}.{n}" if path else n), p

    def parameters(self) -> List[Param]:
        return [p for _, p in self.named_parameters()]

    def get_submodule(self, path: str) -> Module:
        m = self
        for part in [p for p in path.split(".") if p]:
            m = m._modules[part]
        return m

    def set_submodule(self, path: str, new: Module) -> None:
        parent_path, _, name = path.rpartition(".")
        parent = self.get_submodule(parent_path)
        if name not in parent._modules:
            raise KeyError(path)
        parent._modules[name] = new

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode: bool = True) -> Module:
        for m in self.modules():
            object.__setattr__(m, "training", mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    def astype(self, dtype) -> Module:
        for p in self.parameters():
            p.data = p.data.astype(dtype)
            p.grad = None
        return self

    def state_dict(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for path, m in self.named_modules():
            for n, p in m._params.items():
                out[f"{path}.{n}" if path else n] = p.data
            for n, b in m._buffers.items():
                out[f"{path}.{n}" if path else n] = b
        return out

    def load_state_dict(self, sd: Dict[str, np.ndarray]) -> None:
        for key, arr in sd.items():
            path, _, leaf = key.rpartition(".")
            m = self.get_submodule(path)
            if leaf in m._params:
                m._params[leaf].data = np.asarray(arr)
            else:
                m._buffers[leaf] = np.asarray(arr)

    def init_args(self) -> dict:
        return {}

    def spec(self) -> dict:
        return {
            "type": type(self).__name__,
            "args": self.init_args(),
            "children": {n: m.spec() for n, m in self._modules.items()},
        }

    @classmethod
    def from_spec(cls, args: dict, children: Dict[str, Module]) -> Module:
        obj = cls(**args)
        for name, child in children.items():
            obj._modules[name] = child
        return obj

    def own_size_bytes(self) -> int:
        n = sum(p.data.nbytes for p in self._params.values())
        return n + sum(b.nbytes for b in self._buffers.values())

    def size_bytes(self) -> int:
        return sum(m.own_size_bytes() for m in self.modules())

    def num_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}({self.extra_repr()})"]
        for n, m in self._modules.items():
            sub = repr(m).replace("\n", "\n  ")
            lines.append(f"  ({n}): {sub}")
        return "\n".join(lines)

def build_from_spec(spec: dict) -> Module:
    cls = REGISTRY[spec["type"]]
    children = {n: build_from_spec(c) for n, c in spec["children"].items()}
    return cls.from_spec(spec["args"], children)

def _rng(seed):
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)
