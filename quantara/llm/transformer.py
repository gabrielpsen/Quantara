from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantara.errors import CheckpointError
from quantara.nn.layers import Linear
from quantara.nn.module import Module, Param, register, softmax

@dataclass(frozen=True)
class DecoderConfig:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    rms_eps: float = 1e-6
    rope_theta: float = 10000.0
    qkv_bias: bool = True
    vocab_size: int = 0
    num_layers: int = 0

    @classmethod
    def from_hf(cls, cfg: dict) -> DecoderConfig:
        arch = (cfg.get("model_type") or "").lower()
        if arch not in ("llama", "qwen2", "mistral", "qwen3"):
            raise CheckpointError(
                f"unsupported model_type {arch!r} (supported: llama, qwen2, mistral, qwen3)"
            )
        heads = cfg["num_attention_heads"]
        return cls(
            hidden_size=cfg["hidden_size"],
            num_heads=heads,
            num_kv_heads=cfg.get("num_key_value_heads", heads),
            intermediate_size=cfg["intermediate_size"],
            rms_eps=cfg.get("rms_norm_eps", 1e-6),
            rope_theta=float(cfg.get("rope_theta", 10000.0)),
            qkv_bias=arch == "qwen2",
            vocab_size=cfg.get("vocab_size", 0),
            num_layers=cfg.get("num_hidden_layers", 0),
        )

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

@register
class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim, self.eps = int(dim), float(eps)
        self.weight = Param(np.ones(dim, dtype=np.float32))

    def init_args(self):
        return {"dim": self.dim, "eps": self.eps}

    @classmethod
    def from_spec(cls, args, children):
        obj = cls.__new__(cls)
        Module.__init__(obj)
        obj.dim, obj.eps = args["dim"], args["eps"]
        obj.weight = Param(np.empty(0, np.float32))
        return obj

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x * inv * self.weight.data

def rope_tables(seq_len: int, head_dim: int, theta: float, offset: int = 0):
    inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    ang = np.outer(np.arange(offset, offset + seq_len, dtype=np.float64), inv)
    ang = np.concatenate([ang, ang], axis=-1)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    h = x.shape[-1] // 2
    # rotate-half layout, as in llama and qwen2.
    rot = np.concatenate([-x[..., h:], x[..., :h]], axis=-1)
    return x * cos + rot * sin

@register
class GQAttention(Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000.0,
        qkv_bias: bool = True,
        empty: bool = False,
    ):
        super().__init__()
        if hidden_size % num_heads or num_heads % num_kv_heads:
            raise ValueError("hidden_size % heads and heads % kv_heads must be 0")
        self.hidden_size, self.num_heads, self.num_kv_heads = (
            int(hidden_size),
            int(num_heads),
            int(num_kv_heads),
        )
        self.rope_theta, self.qkv_bias = float(rope_theta), bool(qkv_bias)
        self.head_dim = hidden_size // num_heads
        if not empty:
            kv = self.num_kv_heads * self.head_dim
            self.q_proj = Linear(hidden_size, hidden_size, bias=qkv_bias)
            self.k_proj = Linear(hidden_size, kv, bias=qkv_bias)
            self.v_proj = Linear(hidden_size, kv, bias=qkv_bias)
            self.o_proj = Linear(hidden_size, hidden_size, bias=False)

    def init_args(self):
        return {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "rope_theta": self.rope_theta,
            "qkv_bias": self.qkv_bias,
            "empty": True,
        }

    def forward(self, x):
        b, t, _ = x.shape
        h, kv, dh = self.num_heads, self.num_kv_heads, self.head_dim
        q = self.q_proj(x).reshape(b, t, h, dh).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, t, kv, dh).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, t, kv, dh).transpose(0, 2, 1, 3)
        cos, sin = rope_tables(t, dh, self.rope_theta)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        rep = h // kv
        q = q.reshape(b, kv, rep, t, dh)
        att = np.einsum("bgrtd,bgsd->bgrts", q, k, optimize=True) / np.float32(np.sqrt(dh))
        mask = np.triu(np.ones((t, t), dtype=bool), k=1)
        att = np.where(mask, np.float32(-1e30), att)
        out = np.einsum("bgrts,bgsd->bgrtd", softmax(att).astype(np.float32), v, optimize=True)
        out = out.reshape(b, h, t, dh).transpose(0, 2, 1, 3).reshape(b, t, h * dh)
        return self.o_proj(out)

@register
class SwiGLU(Module):
    def __init__(self, hidden_size: int, intermediate_size: int, empty: bool = False):
        super().__init__()
        self.hidden_size, self.intermediate_size = int(hidden_size), int(intermediate_size)
        if not empty:
            self.gate_proj = Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = Linear(intermediate_size, hidden_size, bias=False)

    def init_args(self):
        return {"hidden_size": self.hidden_size, "intermediate_size": self.intermediate_size, "empty": True}

    def forward(self, x):
        g = self.gate_proj(x)
        g = g / (1.0 + np.exp(-g))
        return self.down_proj(g * self.up_proj(x))

@register
class DecoderLayer(Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        rms_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        qkv_bias: bool = True,
        empty: bool = False,
    ):
        super().__init__()
        self.cfg = {
            "hidden_size": int(hidden_size),
            "num_heads": int(num_heads),
            "num_kv_heads": int(num_kv_heads),
            "intermediate_size": int(intermediate_size),
            "rms_eps": float(rms_eps),
            "rope_theta": float(rope_theta),
            "qkv_bias": bool(qkv_bias),
        }
        if not empty:
            self.input_layernorm = RMSNorm(hidden_size, rms_eps)
            self.self_attn = GQAttention(hidden_size, num_heads, num_kv_heads, rope_theta, qkv_bias)
            self.post_attention_layernorm = RMSNorm(hidden_size, rms_eps)
            self.mlp = SwiGLU(hidden_size, intermediate_size)

    def init_args(self):
        return {**self.cfg, "empty": True}

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))

    @classmethod
    def load(cls, ckpt, index: int, config: DecoderConfig) -> DecoderLayer:
        p = f"model.layers.{index}."

        def lin(name, bias):
            w = ckpt.get(p + name + ".weight")
            b = ckpt.get(p + name + ".bias") if bias and (p + name + ".bias") in ckpt else None
            return Linear.from_arrays(w, b)

        layer = cls(
            config.hidden_size,
            config.num_heads,
            config.num_kv_heads,
            config.intermediate_size,
            config.rms_eps,
            config.rope_theta,
            config.qkv_bias,
            empty=True,
        )
        layer.input_layernorm = _norm(ckpt.get(p + "input_layernorm.weight"), config.rms_eps)
        layer.post_attention_layernorm = _norm(
            ckpt.get(p + "post_attention_layernorm.weight"), config.rms_eps
        )
        attn = GQAttention(
            config.hidden_size,
            config.num_heads,
            config.num_kv_heads,
            config.rope_theta,
            config.qkv_bias,
            empty=True,
        )
        attn.q_proj, attn.k_proj, attn.v_proj = (lin(f"self_attn.{n}_proj", True) for n in "qkv")
        attn.o_proj = lin("self_attn.o_proj", False)
        mlp = SwiGLU(config.hidden_size, config.intermediate_size, empty=True)
        mlp.gate_proj, mlp.up_proj, mlp.down_proj = (
            lin(f"mlp.{n}_proj", False) for n in ("gate", "up", "down")
        )
        layer.self_attn, layer.mlp = attn, mlp
        return layer.eval()

def apply_batched(module: Module, h: np.ndarray, batch_size: int) -> np.ndarray:
    return np.concatenate([module(h[i : i + batch_size]) for i in range(0, len(h), batch_size)])

def _norm(weight: np.ndarray, eps: float) -> RMSNorm:
    n = RMSNorm(weight.shape[0], eps)
    n.weight.data = weight.astype(np.float32)
    return n
