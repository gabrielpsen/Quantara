from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import numpy as np

from quantara._version import __version__
from quantara.core.formats import QuantConfig
from quantara.core.tensor import QTensor, quantize_tensor
from quantara.errors import ConfigError
from quantara.llm.checkpoint import Checkpoint
from quantara.llm.transformer import DecoderConfig, DecoderLayer, apply_batched
from quantara.quantize.api import quantize_model
from quantara.quantize.config import PTQConfig
from quantara.serialization import load_model, save_model, save_tensors

log = logging.getLogger("quantara.llm")

EMBED = "model.embed_tokens.weight"
COPIED_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")

@dataclass
class LLMQuantConfig:
    ptq: PTQConfig = field(default_factory=PTQConfig)
    embed_bits: int = 8
    head_bits: int = 8
    io_group_size: int = 128
    calib_from_quantized: bool = True
    batch_size: int = 4

    def __post_init__(self) -> None:
        for n in (self.embed_bits, self.head_bits):
            if n not in (4, 8, 16):
                raise ConfigError("embed_bits / head_bits must be 4, 8 or 16")

    def describe(self) -> dict:
        return {
            "method": self.ptq.method,
            "weight": self.ptq.weight.to_dict(),
            "embed_bits": self.embed_bits,
            "head_bits": self.head_bits,
            "act": None if self.ptq.act is None else self.ptq.act.to_dict(),
            "calib_from_quantized": self.calib_from_quantized,
        }

def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))

def _quantize_rows_streaming(
    get_rows: Callable[[np.ndarray], np.ndarray], shape, bits: int, group: int, chunk_rows: int = 4096
) -> QTensor:
    cfg = QuantConfig(bits=bits, granularity="group", group_size=group)
    codes, scales, zeros, last = [], [], [], None
    for a in range(0, shape[0], chunk_rows):
        last = quantize_tensor(get_rows(np.arange(a, min(a + chunk_rows, shape[0]))), cfg)
        codes.append(last.codes)
        scales.append(last.scale)
        if last.zero is not None:
            zeros.append(last.zero)
    if last is None:
        raise ConfigError("cannot quantize an empty matrix")
    zero = np.concatenate(zeros) if zeros else None
    return QTensor.create(np.concatenate(codes), np.concatenate(scales), zero, cfg, shape, last.pad)

def _save_embeddings_and_head(ckpt: Checkpoint, out_dir: str, cfg: LLMQuantConfig) -> dict:
    vocab, hidden = ckpt.info(EMBED).shape
    head_name = "lm_head.weight" if "lm_head.weight" in ckpt else EMBED

    def convert(name: str, bits: int) -> Union[np.ndarray, QTensor]:
        if bits == 16:
            return ckpt.get(name).astype(np.float16)
        return _quantize_rows_streaming(
            lambda r: ckpt.get_rows(name, r), (vocab, hidden), bits, cfg.io_group_size
        )

    emb = convert(EMBED, cfg.embed_bits)
    save_tensors({"weight": emb}, os.path.join(out_dir, "embed.qnt"))
    tied = head_name == EMBED and cfg.head_bits == cfg.embed_bits
    head = emb if tied else convert(head_name, cfg.head_bits)
    save_tensors({"weight": head, "norm": ckpt.get("model.norm.weight")}, os.path.join(out_dir, "head.qnt"))
    return {"embed_bytes": emb.nbytes, "head_bytes": head.nbytes}

def quantize_llm(
    ckpt_dir: str,
    out_dir: str,
    cfg: LLMQuantConfig,
    calib_tokens: np.ndarray,
    max_layers: Optional[int] = None,
    resume: bool = True,
    progress: Optional[Callable[[dict], None]] = None,
    val_tokens: Optional[np.ndarray] = None,
) -> dict:
    t_start = time.perf_counter()
    os.makedirs(out_dir, exist_ok=True)
    calib_tokens = np.asarray(calib_tokens, dtype=np.int64)
    if calib_tokens.ndim != 2:
        raise ConfigError("calib_tokens must have shape (n_sequences, seq_len)")
    with Checkpoint(ckpt_dir) as ckpt:
        dc = DecoderConfig.from_hf(ckpt.config)
        n_layers = min(dc.num_layers, max_layers) if max_layers else dc.num_layers
        log.info(
            "model: %d layers, hidden=%d, %.2fB params; quantizing %d layer(s) with %s",
            dc.num_layers,
            dc.hidden_size,
            ckpt.num_parameters() / 1e9,
            n_layers,
            cfg.ptq.method,
        )
        bs = cfg.batch_size
        h_fp = ckpt.get_rows(EMBED, calib_tokens)
        h_q = h_fp.copy()
        v_fp = v_q = None
        if val_tokens is not None:
            v_fp = ckpt.get_rows(EMBED, np.asarray(val_tokens, dtype=np.int64))
            v_q = v_fp.copy()
        layer_rows: List[dict] = []
        total_fp = total_q = 0
        for i in range(n_layers):
            t0 = time.perf_counter()
            path = os.path.join(out_dir, f"layer_{i:03d}.qnt")
            fp = DecoderLayer.load(ckpt, i, dc)
            fp_bytes = fp.size_bytes()
            fp_out = apply_batched(fp, h_fp, bs)
            v_fp_out = None if v_fp is None else apply_batched(fp, v_fp, bs)
            row = {}
            if resume and os.path.exists(path):
                ql = load_model(path)
                log.info("layer %d: resumed from %s", i, path)
            else:
                src = h_q if cfg.calib_from_quantized else h_fp
                calib = [src[j : j + bs] for j in range(0, len(src), bs)]
                ql, rep = quantize_model(fp, cfg.ptq, calib=calib, inplace=True)
                save_model(ql, path)
                row["weight_rel_error"] = float(np.mean([r.weight_rel_error for r in rep.layers]))
                row["bits_per_weight"] = float(np.mean([r.bits_per_weight for r in rep.layers]))
            h_fp, v_fp = fp_out, v_fp_out
            h_q = apply_batched(ql, h_q, bs)
            # in-sample, so it flatters gptq. trust val_hidden_rel_error.
            row.update(layer=i, hidden_rel_error=_rel_err(h_q, h_fp), hidden_cosine=_cos(h_q, h_fp))
            if v_fp is not None:
                v_q = apply_batched(ql, v_q, bs)
                row.update(val_hidden_rel_error=_rel_err(v_q, v_fp), val_hidden_cosine=_cos(v_q, v_fp))
            row.update(fp_bytes=fp_bytes, quant_bytes=ql.size_bytes(), seconds=time.perf_counter() - t0)
            total_fp += fp_bytes
            total_q += row["quant_bytes"]
            layer_rows.append(row)
            held_out = f"  held-out rel.err {row['val_hidden_rel_error']:.4f}" if v_fp is not None else ""
            log.info(
                "layer %2d/%d  calib rel.err %.4f  cos %.5f%s  %.1fs",
                i + 1,
                n_layers,
                row["hidden_rel_error"],
                row["hidden_cosine"],
                held_out,
                row["seconds"],
            )
            if progress:
                progress(row)

        io = _save_embeddings_and_head(ckpt, out_dir, cfg)
        for fname in COPIED_FILES:
            if os.path.exists(os.path.join(ckpt_dir, fname)):
                shutil.copy(os.path.join(ckpt_dir, fname), out_dir)
        manifest = {
            "quantara_version": __version__,
            "source": os.path.abspath(ckpt_dir),
            "config": cfg.describe(),
            "decoder": dc.to_dict(),
            "layers_quantized": n_layers,
            "layers": layer_rows,
            "io": io,
            "decoder_fp_bytes": total_fp,
            "decoder_quant_bytes": total_q,
            "seconds": time.perf_counter() - t_start,
        }
        with open(os.path.join(out_dir, "quantara.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    return manifest
