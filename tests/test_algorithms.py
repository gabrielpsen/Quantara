import numpy as np
import pytest

import quantara as qr
from quantara import algorithms as alg
from quantara.core.formats import QuantConfig

def correlated_inputs(n=2048, d=128, rank=16, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, rank))
    a = rng.standard_normal((rank, d))
    return (z @ a + 0.1 * rng.standard_normal((n, d))).astype(np.float32)

def out_err(w, wq, x):
    return float(np.mean((x @ w.T - x @ wq.T) ** 2))

@pytest.mark.parametrize(
    "cfg",
    [
        QuantConfig(bits=3, group_size=32),
        QuantConfig(bits=3, granularity="channel"),
        QuantConfig(bits=3, group_size=32, symmetric=False),
        QuantConfig(format="nf4", group_size=32),
    ],
)
def test_gptq_beats_rtn_on_correlated_inputs(cfg):
    x = correlated_inputs()
    w = np.random.default_rng(1).standard_normal((48, 128)).astype(np.float32) * 0.1
    H = x.T.astype(np.float64) @ x / len(x)
    e_rtn = out_err(w, alg.rtn(w, cfg).dequantize(), x)
    e_gptq = out_err(w, alg.gptq(w, H, cfg).dequantize(), x)
    assert e_gptq < e_rtn

def test_gptq_act_order_and_padding():
    x = correlated_inputs(d=100, seed=2)
    w = np.random.default_rng(3).standard_normal((24, 100)).astype(np.float32) * 0.1
    H = x.T.astype(np.float64) @ x / len(x)
    cfg = QuantConfig(bits=3, group_size=32)
    e_rtn = out_err(w, alg.rtn(w, cfg).dequantize(), x)
    for act_order in (False, True):
        qt = alg.gptq(w, H, cfg, act_order=act_order)
        deq = qt.dequantize()
        assert deq.shape == w.shape
        assert out_err(w, deq, x) < e_rtn

def test_gptq_output_is_valid_grid():
    x = correlated_inputs(d=64, seed=4)
    w = np.random.default_rng(5).standard_normal((8, 64)).astype(np.float32)
    qt = alg.gptq(w, x.T.astype(np.float64) @ x / len(x), QuantConfig(bits=4, group_size=64))
    codes = qt.codes
    assert codes.max() <= 15
    np.testing.assert_allclose(qt.dequantize(), qr.QTensor.from_parts(qt.arrays(), qt.meta()).dequantize())

def test_gptq_handles_dead_inputs():
    x = correlated_inputs(d=64, seed=6)
    x[:, 10] = 0.0
    w = np.random.default_rng(7).standard_normal((8, 64)).astype(np.float32)
    qt = alg.gptq(w, x.T.astype(np.float64) @ x / len(x), QuantConfig(bits=4, group_size=32))
    assert np.isfinite(qt.dequantize()).all()

def test_gptq_rejects_tensor_granularity():
    with pytest.raises(ValueError):
        alg.gptq(np.ones((4, 8), np.float32), np.eye(8), QuantConfig(granularity="tensor"))

def test_awq_never_worse_and_helps_with_outliers():
    rng = np.random.default_rng(8)
    x = rng.standard_normal((1024, 128)).astype(np.float32)
    x[:, rng.choice(128, 4, replace=False)] *= 30.0
    w = rng.standard_normal((64, 128)).astype(np.float32) * 0.1
    cfg = QuantConfig(bits=3, group_size=64)
    e_rtn = out_err(w, alg.rtn(w, cfg).dequantize(), x)
    qt, inv = alg.awq(w, x, cfg)
    e_awq = out_err(w, qt.dequantize() * inv[None, :], x)
    assert e_awq <= e_rtn
    assert e_awq < 0.9 * e_rtn

def test_smoothquant_scales_balance_ranges():
    rng = np.random.default_rng(9)
    w = rng.standard_normal((32, 64)).astype(np.float32)
    amax = np.abs(rng.standard_normal(64)).astype(np.float32) + 0.1
    amax[3] = 100.0
    s = alg.smooth_scales(w, amax, 0.5)
    new_a, new_w = amax / s, np.abs(w).max(axis=0) * s
    assert new_a.max() / new_a.min() < amax.max() / amax.min()
    np.testing.assert_allclose(new_a, new_w, rtol=1e-3)

def test_smoothquant_helps_activation_quantization():
    from quantara import nn

    rng = np.random.default_rng(10)
    lin = nn.Linear(64, 32, seed=1)
    x = rng.standard_normal((512, 64)).astype(np.float32)
    x[:, rng.choice(64, 3, replace=False)] *= 40.0
    ref = lin(x)
    act = qr.ActConfig(bits=6, dynamic=True)
    wcfg = QuantConfig(bits=8, granularity="channel")
    errs = {}
    for method in ("rtn", "smoothquant"):
        q, _ = qr.quantize_model(lin, qr.PTQConfig(weight=wcfg, method=method, act=act), calib=[x])
        errs[method] = qr.metrics.mse(ref, q(x))
    assert errs["smoothquant"] < errs["rtn"]

def test_allocate_bits_respects_budget_and_prefers_sensitive_layers():
    sens = {"a": {2: 100.0, 4: 1.0, 8: 0.9}, "b": {2: 1.0, 4: 0.9, 8: 0.89}}
    numel = {"a": 1000, "b": 1000}
    plan = alg.allocate_bits(sens, numel, 3.0)
    assert plan["a"] == 4 and plan["b"] == 2
    assert sum(plan[n] * numel[n] for n in plan) <= 3.0 * 2000
    plan8 = alg.allocate_bits(sens, numel, 8.0)
    assert plan8 == {"a": 8, "b": 8}
