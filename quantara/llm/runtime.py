from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np

from quantara.core.tensor import QTensor
from quantara.errors import CheckpointError
from quantara.llm.checkpoint import Checkpoint
from quantara.llm.data import load_tokenizer
from quantara.llm.transformer import DecoderConfig, DecoderLayer, RMSNorm, apply_batched
from quantara.serialization import load_model, load_tensors

LOGITS_CHUNK_ROWS = 8192

class CausalLM:
    def __init__(self, path: str):
        self.path = path
        self.quantized = os.path.exists(os.path.join(path, "quantara.json"))
        if self.quantized:
            with open(os.path.join(path, "quantara.json"), encoding="utf-8") as f:
                self.manifest = json.load(f)
            self.config = DecoderConfig(**self.manifest["decoder"])
            self.n_layers = self.manifest["layers_quantized"]
            self._ckpt: Optional[Checkpoint] = None
        else:
            self._ckpt = Checkpoint(path)
            self.config = DecoderConfig.from_hf(self._ckpt.config)
            self.n_layers = self.config.num_layers
            self.manifest = None

    def close(self) -> None:
        if self._ckpt is not None:
            self._ckpt.close()

    def __enter__(self) -> CausalLM:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def ckpt(self) -> Checkpoint:
        if self._ckpt is None:
            raise CheckpointError("this CausalLM wraps a quantized directory, not a float checkpoint")
        return self._ckpt

    def tokenizer(self):
        return load_tokenizer(self.path)

    def _layer(self, i: int):
        if self.quantized:
            return load_model(os.path.join(self.path, f"layer_{i:03d}.qnt"))
        return DecoderLayer.load(self.ckpt, i, self.config)

    def _embed(self, tokens: np.ndarray) -> np.ndarray:
        if self.quantized:
            w = load_tensors(os.path.join(self.path, "embed.qnt"))["weight"]
            return (
                w.dequantize_rows(tokens).astype(np.float32)
                if isinstance(w, QTensor)
                else w[tokens].astype(np.float32)
            )
        return self.ckpt.get_rows("model.embed_tokens.weight", tokens)

    def hidden_states(
        self, tokens: np.ndarray, batch_size: int = 4, n_layers: Optional[int] = None
    ) -> np.ndarray:
        tokens = np.asarray(tokens, dtype=np.int64)
        h = self._embed(tokens)
        for i in range(min(n_layers or self.n_layers, self.n_layers)):
            layer = self._layer(i)
            h = apply_batched(layer, h, batch_size)
            del layer
        return h

    def _head(self):
        if self.quantized:
            head = load_tensors(os.path.join(self.path, "head.qnt"))
            w = head["weight"]

            def rows(a, b):
                if isinstance(w, QTensor):
                    return w.dequantize_rows(np.arange(a, b))
                return w[a:b].astype(np.float32)

            return head["norm"], w.shape[0], rows
        name = "lm_head.weight" if "lm_head.weight" in self.ckpt else "model.embed_tokens.weight"

        def rows(a, b):
            return self.ckpt.get_rows(name, np.arange(a, b))

        return self.ckpt.get("model.norm.weight"), self.ckpt.info(name).shape[0], rows

    def logits(self, tokens: np.ndarray, batch_size: int = 4) -> np.ndarray:
        h = self.hidden_states(tokens, batch_size)
        norm_w, vocab, deq = self._head()
        norm = RMSNorm(self.config.hidden_size, self.config.rms_eps)
        norm.weight.data = np.asarray(norm_w, dtype=np.float32)
        hn = norm(h).reshape(-1, self.config.hidden_size)
        out = np.empty((hn.shape[0], vocab), dtype=np.float32)
        for a in range(0, vocab, LOGITS_CHUNK_ROWS):
            b = min(a + LOGITS_CHUNK_ROWS, vocab)
            out[:, a:b] = hn @ deq(a, b).T
        return out.reshape(*h.shape[:2], vocab)

def next_token_nll(logits: np.ndarray, tokens: np.ndarray) -> Dict[str, float]:
    total, count = 0.0, 0
    for s in range(logits.shape[0]):
        lg = logits[s, :-1].astype(np.float64)
        m = lg.max(axis=-1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(lg - m).sum(axis=-1))
        total += float((lse - lg[np.arange(lg.shape[0]), tokens[s, 1:]]).sum())
        count += lg.shape[0]
    return {"nll": total / count, "perplexity": float(np.exp(total / count)), "tokens": count}

def compare_logits(ref: np.ndarray, test: np.ndarray, tokens: np.ndarray) -> Dict[str, float]:
    r = ref.reshape(-1, ref.shape[-1])
    t = test.reshape(-1, test.shape[-1])
    kl, top1, top5 = 0.0, 0, 0
    for a in range(0, r.shape[0], 256):
        rr, tt = r[a : a + 256].astype(np.float64), t[a : a + 256].astype(np.float64)
        lr = rr - rr.max(axis=-1, keepdims=True)
        lt = tt - tt.max(axis=-1, keepdims=True)
        lpr = lr - np.log(np.exp(lr).sum(axis=-1, keepdims=True))
        lpt = lt - np.log(np.exp(lt).sum(axis=-1, keepdims=True))
        kl += float((np.exp(lpr) * (lpr - lpt)).sum())
        top1 += int((rr.argmax(-1) == tt.argmax(-1)).sum())
        top5 += int((np.argsort(-tt, axis=-1)[:, :5] == rr.argmax(-1)[:, None]).any(axis=-1).sum())
    n = r.shape[0]
    nll_ref, nll_test = next_token_nll(ref, tokens), next_token_nll(test, tokens)
    return {
        "ppl_ref": nll_ref["perplexity"],
        "ppl_test": nll_test["perplexity"],
        "ppl_delta": nll_test["perplexity"] - nll_ref["perplexity"],
        "kl": kl / n,
        "top1_agreement": top1 / n,
        "ref_top1_in_test_top5": top5 / n,
        "tokens": nll_ref["tokens"],
    }
