import copy
import subprocess
import sys

import numpy as np
import pytest

import quantara as qr
from quantara import models
from quantara.core.formats import QuantConfig
from quantara.quantize.layers import QuantConv2d, QuantEmbedding
from quantara.training import Adam, accuracy, fit

@pytest.fixture(scope="module")
def lm():
    m = models.inject_outliers(models.tiny_lm(vocab=96, dim=64, heads=4, layers=2, seed=1))
    rng = np.random.default_rng(0)
    calib = [rng.integers(0, 96, (4, 24)) for _ in range(6)]
    test = [rng.integers(0, 96, (4, 24)) for _ in range(4)]
    return m, calib, test

@pytest.fixture(scope="module")
def trained():
    x, y = models.make_spirals(2100, 3, seed=1)
    xtr, ytr, xte, yte = x[:1500], y[:1500], x[1500:], y[1500:]
    net = models.mlp([2, 96, 96, 3], seed=0)
    fit(net, xtr, ytr, epochs=40, optimizer=Adam(net.parameters(), 3e-3))
    return net, xtr, ytr, xte, yte

def test_int8_is_nearly_lossless(lm):
    m, calib, test = lm
    q, rep = qr.quantize_model(m, qr.PTQConfig(weight=QuantConfig(bits=8, group_size=64)))
    stats = qr.metrics.compare_models(m, q, test)
    assert stats["cosine"] > 0.998 and stats["top1_agreement"] > 0.9
    assert rep.compression > 2.5

@pytest.mark.parametrize("method", ["rtn", "gptq", "awq", "smoothquant"])
def test_all_methods_run_and_keep_quality(lm, method):
    m, calib, test = lm
    cfg = qr.PTQConfig(weight=QuantConfig(bits=4, group_size=32), method=method)
    q, rep = qr.quantize_model(m, cfg, calib=calib)
    stats = qr.metrics.compare_models(m, q, test)
    assert stats["cosine"] > 0.9
    assert any(l.method == method for l in rep.layers)
    assert not any(isinstance(x, qr.nn.Linear) for x in q.modules())

def test_gptq_sequential_and_act_order(lm):
    m, calib, test = lm
    for kw in ({"sequential": True}, {"gptq_act_order": True}):
        q, _ = qr.quantize_model(
            m, qr.PTQConfig(weight=QuantConfig(bits=4, group_size=32), method="gptq", **kw), calib
        )
        assert qr.metrics.compare_models(m, q, test)["cosine"] > 0.9

def test_gptq_not_worse_than_rtn_on_trained_mlp(trained):
    net, xtr, ytr, xte, yte = trained
    calib = [xtr[i : i + 256] for i in range(0, 1024, 256)]
    cfg = QuantConfig(bits=3, group_size=32)
    ref = net(xte)
    errs = {}
    for method in ("rtn", "gptq"):
        q, _ = qr.quantize_model(net, qr.PTQConfig(weight=cfg, method=method), calib)
        errs[method] = qr.metrics.mse(ref, q(xte))
    assert errs["gptq"] < errs["rtn"]

def test_static_and_dynamic_activation_quant(lm):
    m, calib, test = lm
    for dynamic in (True, False):
        cfg = qr.PTQConfig(
            weight=QuantConfig(bits=8, group_size=64), act=qr.ActConfig(bits=8, dynamic=dynamic)
        )
        q, _ = qr.quantize_model(m, cfg, calib=calib)
        assert qr.metrics.compare_models(m, q, test)["cosine"] > (0.99 if dynamic else 0.95)

def test_static_act_requires_calibration(lm):
    m, _, _ = lm
    with pytest.raises(ValueError):
        qr.quantize_model(m, qr.PTQConfig(act=qr.ActConfig(dynamic=False)))
    with pytest.raises(ValueError):
        qr.quantize_model(m, qr.PTQConfig(method="gptq"))

def test_does_not_mutate_original(lm):
    m, calib, _ = lm
    before = {k: v.copy() for k, v in m.state_dict().items()}
    qr.quantize_model(m, qr.PTQConfig(method="awq"), calib)
    for k, v in m.state_dict().items():
        np.testing.assert_array_equal(v, before[k])
    assert any(isinstance(x, qr.nn.Linear) for x in m.modules())

def test_skip_bitmap_and_embeddings(lm):
    m, calib, _ = lm
    cfg = qr.PTQConfig(
        weight=QuantConfig(bits=4, group_size=32),
        skip=("6",),
        bit_map={"*.q_proj": 8, "1.*": 3},
        quantize_embeddings=True,
    )
    q, rep = qr.quantize_model(m, cfg)
    names = {l.name: l for l in rep.layers}
    assert "6" in rep.skipped and "6" not in names
    assert names["1.body.1.q_proj"].config.startswith("int8")
    assert isinstance(q.get_submodule("0"), QuantEmbedding)
    assert isinstance(q.get_submodule("6"), qr.nn.Linear)

def test_conv_model_quantization():
    x, y = models.make_patterns(600, 12, seed=0)
    net = models.small_cnn(1, 4, 12, seed=0)
    fit(net, x[:480], y[:480], epochs=8, batch_size=32, optimizer=Adam(net.parameters(), 3e-3))
    base = accuracy(net, x[480:], y[480:])
    for method in ("rtn", "gptq"):
        q, _ = qr.quantize_model(
            net, qr.PTQConfig(weight=QuantConfig(bits=6, group_size=36), method=method), calib=[x[:64]]
        )
        assert any(isinstance(m, QuantConv2d) for m in q.modules())
        assert accuracy(q, x[480:], y[480:]) >= base - 0.1

def test_mixed_precision_hits_budget(lm):
    m, calib, test = lm
    plan = qr.auto_mixed_precision(
        m, calib[:3], target_avg_bits=4.0, candidates=(2, 4, 8), weight=QuantConfig(bits=4, group_size=32)
    )
    nel = {p: m.get_submodule(p).weight.data.size for p in plan}
    avg = sum(plan[p] * nel[p] for p in plan) / sum(nel.values())
    assert avg <= 4.0 + 1e-9 and len(set(plan.values())) > 1
    q, _ = qr.quantize_model(m, qr.PTQConfig(weight=QuantConfig(bits=4, group_size=32), bit_map=plan))
    assert qr.metrics.compare_models(m, q, test)["cosine"] > 0.9

def test_qat_beats_ptq_at_2_bits(trained):
    net, xtr, ytr, xte, yte = trained
    cfg = QuantConfig(bits=2, group_size=32)
    ptq, _ = qr.quantize_model(net, qr.PTQConfig(weight=cfg))
    acc_ptq = accuracy(ptq, xte, yte)
    qat = copy.deepcopy(net)
    qr.prepare_qat(qat, cfg)
    fit(qat, xtr, ytr, epochs=25, optimizer=Adam(qat.parameters(), 1e-3))
    qr.convert_qat(qat)
    assert all(not isinstance(m, qr.nn.Linear) for m in qat.modules())
    assert accuracy(qat, xte, yte) > acc_ptq + 0.05

def test_qat_with_static_activations(trained):
    net, xtr, ytr, xte, yte = trained
    qat = copy.deepcopy(net)
    qr.prepare_qat(qat, QuantConfig(bits=4, group_size=32), qr.ActConfig(bits=8, dynamic=False))
    fit(qat, xtr, ytr, epochs=6, optimizer=Adam(qat.parameters(), 1e-3))
    qat_acc = accuracy(qat, xte, yte)
    qr.convert_qat(qat)
    assert abs(accuracy(qat, xte, yte) - qat_acc) < 0.05

def test_save_load_roundtrip(lm, tmp_path):
    m, calib, test = lm
    q, _ = qr.quantize_model(
        m,
        qr.PTQConfig(
            weight=QuantConfig(bits=4, group_size=32), method="awq", act=qr.ActConfig(bits=8, dynamic=False)
        ),
        calib,
    )
    path = str(tmp_path / "lm.qnt")
    qr.save_model(q, path)
    q2 = qr.load_model(path)
    for b in test:
        np.testing.assert_allclose(q(b), q2(b), rtol=1e-5, atol=1e-5)
    fp_path = str(tmp_path / "fp.qnt")
    qr.save_model(m, fp_path)
    np.testing.assert_allclose(qr.load_model(fp_path)(test[0]), m(test[0]), rtol=1e-6)

def test_tensor_archive_and_state_dict(tmp_path):
    rng = np.random.default_rng(0)
    weights = {
        "layer.w": rng.standard_normal((64, 128)).astype(np.float32),
        "layer.b": rng.standard_normal(64).astype(np.float32),
        "ids": np.arange(5),
    }
    q = qr.quantize_state_dict(weights, QuantConfig(bits=4, group_size=64))
    assert isinstance(q["layer.w"], qr.QTensor) and not isinstance(q["layer.b"], qr.QTensor)
    path = str(tmp_path / "t.qnt")
    qr.save_tensors(q, path)
    back = qr.load_tensors(path)
    np.testing.assert_array_equal(back["layer.w"].dequantize(), q["layer.w"].dequantize())
    np.testing.assert_array_equal(back["ids"], weights["ids"])
    with pytest.raises(ValueError):
        qr.load_model(path)

def test_size_reduction(lm):
    m, _, _ = lm
    q, rep = qr.quantize_model(m, qr.PTQConfig(weight=QuantConfig(bits=4, group_size=32)))
    assert q.size_bytes() < 0.3 * m.size_bytes()
    assert rep.table()

def test_cli(tmp_path):
    rng = np.random.default_rng(0)
    src, dst, back = (str(tmp_path / n) for n in ("w.npz", "w.qnt", "back.npz"))
    np.savez(src, a=rng.standard_normal((64, 256)).astype(np.float32), b=np.ones(8, np.float32))
    run = lambda *a: subprocess.run([sys.executable, "-m", "quantara", *a], capture_output=True, text=True)
    r = run("quantize", src, dst, "--bits", "4", "--group-size", "64")
    assert r.returncode == 0 and "1/2" in r.stdout, r.stderr
    assert run("inspect", dst).returncode == 0
    assert run("dequantize", dst, back).returncode == 0
    assert np.load(back)["a"].shape == (64, 256)
    assert run("formats").returncode == 0
    assert run("bench", "--size", "128").returncode == 0
    assert run("inspect", str(tmp_path / "missing.qnt")).returncode == 1

def test_load_model_rejects_hostile_archives(tmp_path):
    import io
    import zipfile

    bomb = tmp_path / "bomb.qnt"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as z:
        buf = io.BytesIO()
        np.save(buf, np.zeros(100 << 20, dtype=np.uint8))
        z.writestr("__manifest__.npy", buf.getvalue())
    with pytest.raises(ValueError):
        qr.load_model(str(bomb))
    junk = tmp_path / "junk.qnt"
    junk.write_bytes(b"not a zip")
    with pytest.raises(ValueError):
        qr.load_model(str(junk))

def test_load_model_does_not_allocate_declared_float_layers(tmp_path):
    import json

    args = {"in_features": 10**6, "out_features": 10**6, "bias": True}
    spec = {"type": "Linear", "args": args, "children": {}}
    manifest = {"magic": "quantara", "version": 1, "kind": "model", "spec": spec}
    path = tmp_path / "huge.qnt"
    with open(path, "wb") as f:
        np.savez(f, __manifest__=np.frombuffer(json.dumps(manifest).encode(), np.uint8))
    model = qr.load_model(str(path))
    assert model.in_features == 10**6
