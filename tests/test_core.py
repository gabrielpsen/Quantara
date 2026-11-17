import numpy as np
import pytest

import quantara as qr
from quantara.core.formats import FORMATS, QuantConfig, get_quantizer
from quantara.core.packing import pack, unpack

@pytest.mark.parametrize("bits", range(1, 9))
def test_pack_roundtrip(bits):
    rng = np.random.default_rng(bits)
    codes = rng.integers(0, 2**bits, 1003).astype(np.uint8)
    p = pack(codes, bits)
    assert p.size == (1003 * bits + 7) // 8
    np.testing.assert_array_equal(unpack(p, bits, 1003), codes)

def test_pack_rejects_out_of_range():
    with pytest.raises(ValueError):
        pack(np.array([16], dtype=np.uint8), 4)

@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("sym", [True, False])
def test_int_error_bounded_by_half_step(bits, sym):
    rng = np.random.default_rng(0)
    w = rng.standard_normal((16, 64)).astype(np.float32)
    cfg = QuantConfig(bits=bits, symmetric=sym, granularity="channel", scale_dtype="float32")
    qt = qr.quantize_tensor(w, cfg)
    err = np.abs(qt.dequantize() - w)
    assert np.all(err <= qt.scale.reshape(-1, 1) * 0.5 + 1e-5)

def test_error_decreases_with_bits():
    w = np.random.default_rng(1).standard_normal((32, 128)).astype(np.float32)
    errs = [qr.metrics.mse(w, qr.fake_quant(w, QuantConfig(bits=b, group_size=64))) for b in (2, 3, 4, 6, 8)]
    assert errs == sorted(errs, reverse=True)

@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("gran", ["tensor", "channel", "group"])
def test_all_formats_roundtrip_and_shape(fmt, gran):
    w = np.random.default_rng(2).standard_normal((10, 77)).astype(np.float32)
    cfg = QuantConfig(format=fmt, granularity=gran, group_size=32)
    qt = qr.quantize_tensor(w, cfg)
    d = qt.dequantize()
    assert d.shape == w.shape and np.isfinite(d).all()
    assert qr.metrics.cosine(w, d) > (0.6 if fmt in ("binary", "ternary") else 0.9)
    qt2 = qr.QTensor.from_parts(qt.arrays(), qt.meta())
    np.testing.assert_array_equal(qt2.dequantize(), d)

def test_zero_is_exactly_representable():
    w = np.zeros((4, 32), dtype=np.float32)
    w[0, 0] = 1.0
    for cfg in (
        QuantConfig(bits=4),
        QuantConfig(bits=4, symmetric=False),
        QuantConfig(format="nf4"),
        QuantConfig(format="fp8_e4m3"),
    ):
        assert qr.fake_quant(w, cfg)[1:].max() == 0.0

def test_fp8_tables():
    e4 = get_quantizer(QuantConfig(format="fp8_e4m3")).values
    e5 = get_quantizer(QuantConfig(format="fp8_e5m2")).values
    assert np.abs(e4).max() == 448.0 and np.abs(e5).max() == 57344.0

def test_finite_on_constant_and_tiny_inputs():
    for w in (
        np.ones((4, 16), np.float32),
        np.full((4, 16), 1e-9, np.float32),
        np.zeros((4, 16), np.float32),
    ):
        for cfg in (QuantConfig(bits=4), QuantConfig(bits=4, symmetric=False), QuantConfig(format="nf4")):
            assert np.isfinite(qr.fake_quant(w, cfg)).all()

def test_axis_and_conv_layout():
    w = np.random.default_rng(3).standard_normal((6, 3, 3, 3)).astype(np.float32)
    for axis in (0, 1, -1):
        cfg = QuantConfig(bits=8, granularity="channel", axis=axis)
        qt = qr.quantize_tensor(w, cfg)
        assert qt.scale.shape[0] == w.shape[axis]
        assert qr.metrics.sqnr_db(w, qt.dequantize()) > 35

def test_observers_mse_beats_minmax_with_outliers():
    rng = np.random.default_rng(4)
    w = rng.standard_normal((32, 256)).astype(np.float32)
    w[:, 0] = 8.0
    base = QuantConfig(bits=3, granularity="channel")
    e_minmax = qr.metrics.mse(w, qr.fake_quant(w, base))
    e_mse = qr.metrics.mse(w, qr.fake_quant(w, base.with_(observer="mse")))
    e_pct = qr.metrics.mse(w, qr.fake_quant(w, base.with_(observer="percentile", percentile=99.0)))
    assert e_mse < e_minmax
    assert np.isfinite(e_pct)

def test_size_accounting():
    w = np.random.default_rng(5).standard_normal((64, 128)).astype(np.float32)
    qt = qr.quantize_tensor(w, QuantConfig(bits=4, group_size=64))
    assert qt.nbytes == 64 * 128 // 2 + (64 * 2) * 2
    assert abs(qt.bits_per_weight - 4.25) < 1e-9

def test_dequantize_rows_matches_full():
    w = np.random.default_rng(6).standard_normal((50, 70)).astype(np.float32)
    qt = qr.quantize_tensor(w, QuantConfig(bits=4, group_size=32, symmetric=False))
    rows = np.array([[3, 7], [49, 0]])
    np.testing.assert_allclose(qt.dequantize_rows(rows), qt.dequantize()[rows])

def test_invalid_config():
    with pytest.raises(ValueError):
        QuantConfig(bits=1)
    with pytest.raises(ValueError):
        QuantConfig(format="nope")
