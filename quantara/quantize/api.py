from __future__ import annotations

import copy
import fnmatch
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from quantara import algorithms as alg
from quantara.core.formats import QuantConfig
from quantara.core.observers import ActConfig, static_params
from quantara.core.tensor import quantize_tensor
from quantara.nn.layers import Conv2d, Embedding, Linear
from quantara.nn.module import Module
from quantara.quantize.calibrate import collect_stats, LayerStats
from quantara.quantize.config import PTQConfig
from quantara.quantize.layers import QuantConv2d, QuantEmbedding, QuantLinear
from quantara.quantize.report import LayerReport, QuantReport

def _targets(model: Module, cfg: PTQConfig) -> Tuple[List[Tuple[str, Module]], List[str]]:
    kinds = (Linear, Conv2d) + ((Embedding,) if cfg.quantize_embeddings else ())
    out, skipped = [], []
    for path, m in model.named_modules():
        if not isinstance(m, kinds):
            continue
        if any(fnmatch.fnmatch(path, p) for p in cfg.skip) or m.weight.data.size < cfg.min_numel:
            skipped.append(path or "<root>")
            continue
        out.append((path, m))
    return out, skipped

def _static_act(st: Optional[LayerStats], act: ActConfig, inv_scale: Optional[np.ndarray]):
    if st is None:
        raise RuntimeError("static activation quantization needs calibration data that reaches this layer")
    inv = 1.0 if inv_scale is None else inv_scale
    if act.observer == "percentile" and st.samples is not None:
        x = st.samples * inv
        mn = float(np.percentile(x, 100 - act.percentile))
        mx = float(np.percentile(x, act.percentile))
    else:
        mn, mx = float((st.chmin * inv).min()), float((st.chmax * inv).max())
    return static_params(mn, mx, act.bits, act.symmetric)

def _quantize_layer(path: str, mod: Module, cfg: QuantConfig, st: Optional[LayerStats], pc: PTQConfig):
    w = mod.weight.data.astype(np.float32)
    method = pc.method
    in_scale = None
    out_mse = None

    if isinstance(mod, Embedding):
        qt = quantize_tensor(w, cfg)
        new = QuantEmbedding.from_float(qt, cache_weights=False)
        used = "rtn"
    else:
        is_linear = isinstance(mod, Linear)
        w2 = w.reshape(w.shape[0], -1)
        if method == "gptq" and st is not None and st.H is not None:
            qt = alg.gptq(w2, st.H, cfg, pc.gptq_blocksize, pc.gptq_percdamp, pc.gptq_act_order)
            qt.shape = tuple(w.shape)
            used = "gptq"
        elif method == "awq" and is_linear and st is not None and st.samples is not None:
            qt, in_scale = alg.awq(w, st.samples, cfg, pc.awq_grid)
            used = "awq"
        elif method == "smoothquant" and is_linear and st is not None:
            s = alg.smooth_scales(w, st.absmax, pc.smooth_alpha)
            qt = quantize_tensor(w * s[None, :], cfg)
            in_scale = (1.0 / s).astype(np.float32)
            used = "smoothquant"
        else:
            qt = quantize_tensor(w, cfg)
            used = "rtn"
        bias = mod.bias.data.astype(np.float32) if mod.has_bias else None
        act_params = None
        if pc.act is not None and not pc.act.dynamic:
            act_params = _static_act(st, pc.act, in_scale)
        if is_linear:
            new = QuantLinear.from_float(qt, bias, in_scale, pc.act, act_params, pc.cache_weights)
        else:
            new = QuantConv2d.from_float(
                qt, bias, mod.stride, mod.padding, pc.act, act_params, pc.cache_weights
            )

    deq = qt.dequantize().reshape(w.shape)
    if in_scale is not None:
        deq = deq * in_scale.reshape((1, -1) + (1,) * (w.ndim - 2))
    rel = float(np.linalg.norm(deq - w) / max(np.linalg.norm(w), 1e-12))
    if st is not None and st.samples is not None and isinstance(mod, Linear):
        x = st.samples
        out_mse = float(np.mean((x @ deq.T - x @ w.T) ** 2))
    rep = LayerReport(
        path,
        type(mod).__name__,
        tuple(w.shape),
        cfg.name,
        used,
        qt.bits_per_weight,
        w.size * 4,
        qt.nbytes,
        rel,
        out_mse,
    )
    return new, rep

def quantize_model(
    model: Module, config: Optional[PTQConfig] = None, calib: Optional[Iterable] = None, inplace: bool = False
) -> Tuple[Module, QuantReport]:
    pc = config or PTQConfig()
    qmodel = model if inplace else copy.deepcopy(model)
    qmodel.eval()
    targets, skipped = _targets(qmodel, pc)
    static_act = pc.act is not None and not pc.act.dynamic
    needs = pc.method != "rtn" or static_act
    batches = None
    if needs:
        if calib is None:
            raise ValueError(
                f"method {pc.method!r}{' with static activations' if static_act else ''} "
                "requires calibration data (calib=...)"
            )
        batches = list(calib)
        if not batches:
            raise ValueError("calibration data is empty")

    stats: Dict[str, LayerStats] = {}
    seq = pc.method == "gptq" and pc.sequential
    if needs and not seq:
        lin_targets = {p: m for p, m in targets if not isinstance(m, Embedding)}
        stats = collect_stats(
            qmodel, batches, lin_targets, hessian=pc.method == "gptq", max_samples=pc.calib_samples
        )

    report = QuantReport(skipped=skipped)
    for path, mod in targets:
        cfg = pc.cfg_for(path)
        st = stats.get(path)
        if seq and not isinstance(mod, Embedding):
            st = collect_stats(qmodel, batches, {path: mod}, hessian=True, max_samples=pc.calib_samples).get(
                path
            )
        new, rep = _quantize_layer(path, mod, cfg, st, pc)
        if path:
            qmodel.set_submodule(path, new)
        else:
            qmodel = new
        report.layers.append(rep)
        report.original_bytes += rep.original_bytes
        report.quantized_bytes += rep.quantized_bytes
    kept = qmodel.size_bytes() - sum(r.quantized_bytes for r in report.layers)
    report.quantized_bytes += max(kept, 0)
    report.original_bytes += max(kept, 0)
    return qmodel, report

def sensitivity_analysis(
    model: Module,
    calib: Iterable,
    candidates: Sequence[int] = (2, 3, 4, 8),
    weight: Optional[QuantConfig] = None,
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, int]]:
    base = weight or QuantConfig(bits=4, group_size=64)
    batches = list(calib)
    model = copy.deepcopy(model).eval()
    ref = [np.asarray(model(b), dtype=np.float64) for b in batches]
    pc0 = PTQConfig(weight=base)
    targets, _ = _targets(model, pc0)
    sens: Dict[str, Dict[int, float]] = {}
    numel: Dict[str, int] = {}
    for path, mod in targets:
        if not path:
            continue
        sens[path], numel[path] = {}, mod.weight.data.size
        for bits in candidates:
            cfg = base.with_(bits=bits, format="int")
            new, _ = _quantize_layer(path, mod, cfg, None, PTQConfig(weight=cfg))
            model.set_submodule(path, new)
            err = np.mean(
                [np.mean((np.asarray(model(b), np.float64) - r) ** 2) for b, r in zip(batches, ref)]
            )
            sens[path][bits] = float(err)
            model.set_submodule(path, mod)
    return sens, numel

def auto_mixed_precision(
    model: Module,
    calib: Iterable,
    target_avg_bits: float = 4.0,
    candidates: Sequence[int] = (2, 3, 4, 8),
    weight: Optional[QuantConfig] = None,
) -> Dict[str, int]:
    sens, numel = sensitivity_analysis(model, calib, candidates, weight)
    return alg.allocate_bits(sens, numel, target_avg_bits)
