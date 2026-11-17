from __future__ import annotations

import copy
import time

import numpy as np

from quantara import metrics, models
from quantara._version import __version__
from quantara.core.formats import FORMATS, QuantConfig
from quantara.core.tensor import QTensor, quantize_tensor
from quantara.quantize.api import quantize_model
from quantara.quantize.config import PTQConfig
from quantara.serialization import (
    _read,
    load_model,
    load_tensors,
    quantize_state_dict,
    save_model,
    save_tensors,
)
from quantara.training.loop import accuracy, fit
from quantara.training.optim import Adam
from quantara.training.qat import convert_qat, prepare_qat

def _cfg_from_args(a) -> QuantConfig:
    return QuantConfig(
        bits=a.bits,
        format=a.format,
        symmetric=not a.asymmetric,
        granularity=a.granularity,
        group_size=a.group_size,
        observer=a.observer,
    )

def _add_quant_args(p):
    p.add_argument("--bits", type=int, default=4, help="bits for the int format (2-8)")
    p.add_argument("--format", choices=FORMATS, default="int")
    p.add_argument("--granularity", choices=("tensor", "channel", "group"), default="group")
    p.add_argument("--group-size", type=int, default=128)
    p.add_argument("--asymmetric", action="store_true", help="asymmetric integer grid")
    p.add_argument("--observer", choices=("minmax", "percentile", "mse"), default="minmax")

def cmd_formats(_a) -> int:
    print("Formats:")
    info = {
        "int": "uniform integer (2-8 bits), symmetric or asymmetric",
        "nf4": "NormalFloat 4-bit (QLoRA)",
        "fp4": "float 4-bit E2M1",
        "fp8_e4m3": "float 8-bit E4M3",
        "fp8_e5m2": "float 8-bit E5M2",
        "binary": "1 bit (+/- scale)",
        "ternary": "2 bits {-1, 0, +1} (TWN)",
    }
    for f in FORMATS:
        print(f"  {f:<10} {info[f]}")
    print("\nGranularity: tensor, channel, group | Observers: minmax, percentile, mse")
    print("PTQ: rtn, gptq, awq, smoothquant | QAT: straight-through estimator")
    return 0

def cmd_demo(a) -> int:
    print(f"Quantara {__version__} - demo: 2D spirals, MLP\n")
    x, y = models.make_spirals(2400, 3, seed=1)
    xtr, ytr, xte, yte = x[:1800], y[:1800], x[1800:], y[1800:]
    net = models.mlp([2, 128, 128, 64, 3], seed=0)
    print("Training float32 model...")
    fit(net, xtr, ytr, epochs=a.epochs, batch_size=64, optimizer=Adam(net.parameters(), 3e-3))
    base = accuracy(net, xte, yte)
    calib = [xtr[i : i + 256] for i in range(0, 1024, 256)]
    rows = {"float32": {"acc": base, "MB": net.size_bytes() / 1e6, "x": 1.0}}

    def run(name, cfg, method="rtn", **kw):
        q, rep = quantize_model(net, PTQConfig(weight=cfg, method=method, **kw), calib)
        rows[name] = {"acc": accuracy(q, xte, yte), "MB": q.size_bytes() / 1e6, "x": rep.compression}

    g = 32
    run("int8 rtn", QuantConfig(bits=8, group_size=g))
    run("int4 rtn", QuantConfig(bits=4, group_size=g))
    run("int4 gptq", QuantConfig(bits=4, group_size=g), "gptq")
    run("int3 rtn", QuantConfig(bits=3, group_size=g))
    run("int3 gptq", QuantConfig(bits=3, group_size=g), "gptq", sequential=True)
    run("int3 awq", QuantConfig(bits=3, group_size=g), "awq")
    run("nf4 rtn", QuantConfig(format="nf4", group_size=g))
    run("int2 rtn", QuantConfig(bits=2, group_size=g))
    run("int2 gptq", QuantConfig(bits=2, group_size=g), "gptq")

    print("Training with QAT (int2)...")
    qat_net = copy.deepcopy(net)
    prepare_qat(qat_net, QuantConfig(bits=2, group_size=g))
    fit(
        qat_net,
        xtr,
        ytr,
        epochs=max(a.epochs // 2, 5),
        batch_size=64,
        optimizer=Adam(qat_net.parameters(), 1e-3),
    )
    convert_qat(qat_net)
    rows["int2 QAT"] = {
        "acc": accuracy(qat_net, xte, yte),
        "MB": qat_net.size_bytes() / 1e6,
        "x": net.size_bytes() / qat_net.size_bytes(),
    }
    print()
    print(metrics.format_table(rows, ["acc", "MB", "x"]))
    if a.output:
        save_model(qat_net, a.output)
        print(f"\nmodel saved to {a.output}")
    return 0

def cmd_bench(a) -> int:
    rng = np.random.default_rng(0)
    w = rng.standard_normal((a.size, a.size)).astype(np.float32) * 0.02
    w[:, rng.choice(a.size, max(1, a.size // 100), replace=False)] *= 12
    print(f"{a.size}x{a.size} gaussian matrix with outlier channels (SQNR in dB, higher is better)\n")
    rows = {}
    benchmarks = [("int", 8), ("int", 6), ("int", 4), ("int", 3), ("int", 2), ("nf4", 4), ("fp4", 4),
                  ("fp8_e4m3", 8), ("fp8_e5m2", 8), ("ternary", 2), ("binary", 1)]
    for fmt, bits in benchmarks:
        for gran, gs in (("channel", 0), ("group", a.group_size)):
            cfg = QuantConfig(bits=bits, format=fmt, granularity=gran, group_size=max(gs, 1))
            t0 = time.perf_counter()
            qt = quantize_tensor(w, cfg)
            dt = (time.perf_counter() - t0) * 1e3
            deq = qt.dequantize()
            rows[cfg.name] = {
                "sqnr_db": metrics.sqnr_db(w, deq),
                "bits/w": qt.bits_per_weight,
                "ratio": 32 / qt.bits_per_weight,
                "ms": dt,
            }
    print(metrics.format_table(rows))
    return 0

def cmd_quantize(a) -> int:
    data = np.load(a.input, allow_pickle=False)
    weights = {k: data[k] for k in data.files}
    q = quantize_state_dict(weights, _cfg_from_args(a), min_numel=a.min_numel)
    save_tensors(q, a.output)
    orig = sum(w.nbytes for w in weights.values())
    new = sum(t.nbytes for t in q.values())
    nq = sum(isinstance(t, QTensor) for t in q.values())
    print(
        f"{nq}/{len(q)} tensors quantized  |  {orig / 1e6:.2f} MB -> {new / 1e6:.2f} MB "
        f"({orig / max(new, 1):.2f}x)  ->  {a.output}"
    )
    return 0

def cmd_inspect(a) -> int:
    manifest, _ = _read(a.input)
    print(f"{a.input}: Quantara archive v{manifest['version']} ({manifest['kind']})")
    if manifest["kind"] == "tensors":
        total_o = total_n = 0
        for name, t in load_tensors(a.input).items():
            if isinstance(t, QTensor):
                print(f"  {name:<40}{t.shape!s:<18}{t.cfg.name:<20}{t.bits_per_weight:6.2f} bpw")
                total_o += t.numel * 4
                total_n += t.nbytes
            else:
                print(f"  {name:<40}{t.shape!s:<18}{'float (kept)':<20}")
                total_o += t.nbytes
                total_n += t.nbytes
        print(f"total: {total_o / 1e6:.3f} MB -> {total_n / 1e6:.3f} MB ({total_o / max(total_n, 1):.2f}x)")
    else:
        m = load_model(a.input)
        print(m)
        print(f"size: {m.size_bytes() / 1e6:.3f} MB")
    return 0

def cmd_dequantize(a) -> int:
    out = {k: (t.dequantize() if isinstance(t, QTensor) else t) for k, t in load_tensors(a.input).items()}
    np.savez(a.output, **out)
    print(f"{len(out)} float32 tensors -> {a.output}")
    return 0
