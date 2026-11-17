from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

from quantara.nn.module import Module, softmax

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2))

def sqnr_db(ref: np.ndarray, approx: np.ndarray) -> float:
    ref = np.asarray(ref, np.float64)
    noise = np.sum((ref - np.asarray(approx, np.float64)) ** 2)
    if noise == 0:
        return float("inf")
    return float(10.0 * np.log10(np.sum(ref**2) / noise))

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))

def kl_divergence(logits_ref: np.ndarray, logits_q: np.ndarray) -> float:
    p = softmax(np.asarray(logits_ref, np.float64).reshape(-1, logits_ref.shape[-1]))
    q = softmax(np.asarray(logits_q, np.float64).reshape(-1, logits_q.shape[-1]))
    return float(np.mean(np.sum(p * (np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(q, 1e-12))), axis=-1)))

def top1_agreement(logits_ref: np.ndarray, logits_q: np.ndarray) -> float:
    return float(np.mean(logits_ref.argmax(axis=-1) == logits_q.argmax(axis=-1)))

def compare_models(ref: Module, quant: Module, batches: Iterable) -> Dict[str, float]:
    ref.eval(), quant.eval()
    r, q = [], []
    for b in batches:
        r.append(np.asarray(ref(b), np.float64))
        q.append(np.asarray(quant(b), np.float64))
    r = np.concatenate([a.reshape(-1, a.shape[-1]) for a in r])
    q = np.concatenate([a.reshape(-1, a.shape[-1]) for a in q])
    return {
        "mse": mse(r, q),
        "sqnr_db": sqnr_db(r, q),
        "cosine": cosine(r, q),
        "kl": kl_divergence(r, q),
        "top1_agreement": top1_agreement(r, q),
        "size_ref_mb": ref.size_bytes() / 1e6,
        "size_quant_mb": quant.size_bytes() / 1e6,
        "compression": ref.size_bytes() / max(quant.size_bytes(), 1),
    }

def format_table(rows: Dict[str, Dict[str, float]], columns: Optional[list] = None) -> str:
    if not rows:
        return ""
    cols = columns or list(next(iter(rows.values())).keys())
    w0 = max(len(k) for k in rows) + 2
    cw = max(14, max(len(c) for c in cols) + 2)
    head = f"{'':<{w0}}" + "".join(f"{c:>{cw}}" for c in cols)
    out = [head, "-" * len(head)]
    for name, vals in rows.items():
        cells = []
        for c in cols:
            v = vals.get(c, "")
            cells.append(f"{v:>{cw}.5g}" if isinstance(v, (int, float, np.floating)) else f"{v!s:>{cw}}")
        out.append(f"{name:<{w0}}" + "".join(cells))
    return "\n".join(out)
