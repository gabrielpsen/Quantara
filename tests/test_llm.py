import json
import os
import struct

import numpy as np
import pytest

import quantara as qr
from quantara import llm
from quantara.llm.pipeline import _quantize_rows_streaming
from quantara.llm.checkpoint import Checkpoint, SafetensorsFile, write_safetensors
from quantara.errors import CheckpointError, ConfigError
from quantara.core.formats import QuantConfig
from quantara.llm.transformer import DecoderConfig, DecoderLayer, GQAttention, apply_rope, rope_tables

H, HEADS, KV, FFN, V, LAYERS = 64, 4, 2, 128, 100, 2

def make_checkpoint(path, sharded=True, seed=0):
    rng = np.random.default_rng(seed)
    w = lambda *s: (rng.standard_normal(s) * 0.05).astype(np.float32)
    t = {"model.embed_tokens.weight": w(V, H), "model.norm.weight": 1 + w(H), "lm_head.weight": w(V, H)}
    kvd = KV * (H // HEADS)
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        t.update(
            {
                p + "input_layernorm.weight": 1 + w(H),
                p + "post_attention_layernorm.weight": 1 + w(H),
                p + "self_attn.q_proj.weight": w(H, H),
                p + "self_attn.q_proj.bias": w(H),
                p + "self_attn.k_proj.weight": w(kvd, H),
                p + "self_attn.k_proj.bias": w(kvd),
                p + "self_attn.v_proj.weight": w(kvd, H),
                p + "self_attn.v_proj.bias": w(kvd),
                p + "self_attn.o_proj.weight": w(H, H),
                p + "mlp.gate_proj.weight": w(FFN, H),
                p + "mlp.up_proj.weight": w(FFN, H),
                p + "mlp.down_proj.weight": w(H, FFN),
            }
        )
    os.makedirs(path, exist_ok=True)
    cfg = {
        "model_type": "qwen2",
        "hidden_size": H,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV,
        "intermediate_size": FFN,
        "num_hidden_layers": LAYERS,
        "vocab_size": V,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
    }
    json.dump(cfg, open(os.path.join(path, "config.json"), "w"))
    names = list(t)
    if sharded:
        a, b = names[: len(names) // 2], names[len(names) // 2 :]
        write_safetensors({k: t[k] for k in a}, os.path.join(path, "m-1.safetensors"))
        write_safetensors({k: t[k] for k in b}, os.path.join(path, "m-2.safetensors"))
        wm = dict.fromkeys(a, "m-1.safetensors") | dict.fromkeys(b, "m-2.safetensors")
        json.dump({"weight_map": wm}, open(os.path.join(path, "model.safetensors.index.json"), "w"))
    else:
        write_safetensors(t, os.path.join(path, "model.safetensors"))
    return path, t

@pytest.fixture(scope="module")
def ckpt_dir(tmp_path_factory):
    return make_checkpoint(str(tmp_path_factory.mktemp("tiny")))[0]

def tokens(n=4, t=24, seed=1):
    return np.random.default_rng(seed).integers(0, V, (n, t))

def test_safetensors_roundtrip_and_rows(tmp_path):
    a = np.random.default_rng(0).standard_normal((10, 7)).astype(np.float32)
    write_safetensors({"a": a, "b": np.arange(5, dtype=np.int64)}, str(tmp_path / "x.safetensors"))
    f = SafetensorsFile(str(tmp_path / "x.safetensors"))
    np.testing.assert_array_equal(f.get("a"), a)
    np.testing.assert_array_equal(f.get_rows("a", np.array([[3, 1], [9, 0]])), a[[[3, 1], [9, 0]]])
    assert f.tensors["b"].shape == (5,)
    f.close()

def test_bf16_conversion(tmp_path):
    vals = np.array([1.0, -2.5, 0.15625, 3.0e4], dtype=np.float32)
    raw = (vals.view(np.uint32) >> 16).astype(np.uint16)
    header = json.dumps({"w": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]}}).encode()
    header += b" " * ((8 - len(header) % 8) % 8)
    p = str(tmp_path / "bf.safetensors")
    with open(p, "wb") as fh:
        fh.write(struct.pack("<Q", len(header)) + header + raw.tobytes())
    f = SafetensorsFile(p)
    np.testing.assert_allclose(f.get("w"), vals, rtol=1e-2)
    f.close()

def test_truncated_file_is_rejected(tmp_path):
    p = str(tmp_path / "x.safetensors")
    write_safetensors({"a": np.ones((100, 100), np.float32)}, p)
    data = open(p, "rb").read()
    open(p, "wb").write(data[: len(data) // 2])
    with pytest.raises(CheckpointError):
        SafetensorsFile(p)

def test_checkpoint_dir_errors(tmp_path):
    with pytest.raises(CheckpointError):
        Checkpoint(str(tmp_path / "missing"))
    with pytest.raises(CheckpointError):
        Checkpoint(str(tmp_path))

def test_sharded_checkpoint(ckpt_dir):
    with Checkpoint(ckpt_dir) as ck:
        assert "lm_head.weight" in ck and ck.info("lm_head.weight").shape == (V, H)
        assert ck.num_parameters() > 0
        with pytest.raises(CheckpointError):
            ck.info("nope")

def test_rope_properties():
    dh = 16
    cos, sin = rope_tables(32, dh, 10000.0)
    x = np.random.default_rng(0).standard_normal((1, 1, 32, dh)).astype(np.float32)
    r = apply_rope(x, cos, sin)
    np.testing.assert_allclose(np.linalg.norm(r, axis=-1), np.linalg.norm(x, axis=-1), rtol=1e-5)
    np.testing.assert_allclose(r[..., 0, :], x[..., 0, :], atol=1e-6)
    q = np.random.default_rng(1).standard_normal(dh).astype(np.float32)
    k = np.random.default_rng(2).standard_normal(dh).astype(np.float32)
    qq = apply_rope(np.tile(q, (1, 1, 32, 1)), cos, sin)[0, 0]
    kk = apply_rope(np.tile(k, (1, 1, 32, 1)), cos, sin)[0, 0]
    assert qq[5] @ kk[3] == pytest.approx(qq[9] @ kk[7], rel=1e-4)

def test_attention_is_causal_and_gqa_shapes():
    att = GQAttention(H, HEADS, KV).eval()
    x = np.random.default_rng(0).standard_normal((2, 10, H)).astype(np.float32)
    y = att(x)
    x2 = x.copy()
    x2[:, 7:] += 3.0
    np.testing.assert_allclose(att(x2)[:, :7], y[:, :7], atol=1e-5)
    assert y.shape == x.shape and att.k_proj.out_features == KV * (H // HEADS)

def test_decoder_config_validation():
    with pytest.raises(CheckpointError):
        DecoderConfig.from_hf({"model_type": "gpt2"})
    assert (
        DecoderConfig.from_hf(
            {"model_type": "llama", "hidden_size": 8, "num_attention_heads": 2, "intermediate_size": 16}
        ).num_kv_heads
        == 2
    )

def test_decoder_layer_load_matches_reference(ckpt_dir):
    with Checkpoint(ckpt_dir) as ck:
        dc = DecoderConfig.from_hf(ck.config)
        layer = DecoderLayer.load(ck, 0, dc)
        x = np.random.default_rng(0).standard_normal((2, 12, H)).astype(np.float32)
        y = layer(x)
        g = lambda n: ck.get(f"model.layers.0.{n}")
        rms = lambda a, w: a / np.sqrt((a * a).mean(-1, keepdims=True) + 1e-6) * w
        hn = rms(x, g("input_layernorm.weight"))
        q = hn @ g("self_attn.q_proj.weight").T + g("self_attn.q_proj.bias")
        k = hn @ g("self_attn.k_proj.weight").T + g("self_attn.k_proj.bias")
        v = hn @ g("self_attn.v_proj.weight").T + g("self_attn.v_proj.bias")
        b, t, dh = 2, 12, H // HEADS
        q = q.reshape(b, t, HEADS, dh).transpose(0, 2, 1, 3)
        k = k.reshape(b, t, KV, dh).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, KV, dh).transpose(0, 2, 1, 3)
        cos, sin = rope_tables(t, dh, 10000.0)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k, v = np.repeat(k, HEADS // KV, axis=1), np.repeat(v, HEADS // KV, axis=1)
        s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh)
        s = np.where(np.triu(np.ones((t, t), bool), 1), -1e30, s)
        p = np.exp(s - s.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        a = (p @ v).transpose(0, 2, 1, 3).reshape(b, t, H) @ g("self_attn.o_proj.weight").T
        x1 = x + a
        hn2 = rms(x1, g("post_attention_layernorm.weight"))
        gate = hn2 @ g("mlp.gate_proj.weight").T
        ref = (
            x1
            + ((gate / (1 + np.exp(-gate))) * (hn2 @ g("mlp.up_proj.weight").T)) @ g("mlp.down_proj.weight").T
        )
        np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4)

def test_quantized_decoder_layer_serializes(ckpt_dir, tmp_path):
    with Checkpoint(ckpt_dir) as ck:
        layer = DecoderLayer.load(ck, 1, DecoderConfig.from_hf(ck.config))
    x = np.random.default_rng(0).standard_normal((2, 9, H)).astype(np.float32)
    q, _ = qr.quantize_model(layer, qr.PTQConfig(weight=QuantConfig(bits=8, group_size=32)))
    qr.save_model(q, str(tmp_path / "l.qnt"))
    q2 = qr.load_model(str(tmp_path / "l.qnt"))
    np.testing.assert_allclose(q(x), q2(x), rtol=1e-5, atol=1e-5)
    assert qr.metrics.cosine(layer(x), q(x)) > 0.999

def test_quantize_llm_end_to_end(ckpt_dir, tmp_path):
    out = str(tmp_path / "q")
    calib = tokens(8, 24, seed=3)
    cfg = llm.LLMQuantConfig(
        ptq=qr.PTQConfig(weight=QuantConfig(bits=8, group_size=32), method="gptq", cache_weights=False),
        embed_bits=8,
        head_bits=8,
    )
    m = llm.quantize_llm(ckpt_dir, out, cfg, calib, val_tokens=tokens(4, 24, seed=11))
    assert all(0 <= r["val_hidden_rel_error"] < 0.1 for r in m["layers"])
    assert m["layers_quantized"] == LAYERS and m["decoder_quant_bytes"] < m["decoder_fp_bytes"]
    assert all(r["hidden_cosine"] > 0.99 for r in m["layers"])
    ev = tokens(4, 24, seed=9)
    with llm.CausalLM(ckpt_dir) as ref, llm.CausalLM(out) as q:
        assert q.quantized and not ref.quantized
        lr, lq = ref.logits(ev), q.logits(ev)
        assert lr.shape == (4, 24, V)
        stats = llm.compare_logits(lr, lq, ev)
    assert stats["top1_agreement"] > 0.9 and stats["kl"] < 0.05
    assert abs(stats["ppl_delta"]) < 0.05 * stats["ppl_ref"]
    m2 = llm.quantize_llm(ckpt_dir, out, cfg, calib, resume=True)
    assert m2["layers_quantized"] == LAYERS

def test_quantize_llm_low_bit_methods_run(ckpt_dir, tmp_path):
    calib = tokens(8, 24, seed=4)
    for method in ("rtn", "awq"):
        cfg = llm.LLMQuantConfig(
            ptq=qr.PTQConfig(weight=QuantConfig(bits=4, group_size=32), method=method, cache_weights=False),
            embed_bits=16,
            head_bits=4,
        )
        m = llm.quantize_llm(ckpt_dir, str(tmp_path / method), cfg, calib, max_layers=1)
        assert m["layers_quantized"] == 1
        with llm.CausalLM(str(tmp_path / method)) as q:
            assert np.isfinite(q.logits(tokens(2, 16))).all()

def test_llm_input_validation(ckpt_dir, tmp_path):
    with pytest.raises(ConfigError):
        llm.LLMQuantConfig(embed_bits=3)
    with pytest.raises(ConfigError):
        llm.quantize_llm(ckpt_dir, str(tmp_path / "o"), llm.LLMQuantConfig(), np.arange(10))

def test_compare_logits_identity():
    lg = np.random.default_rng(0).standard_normal((2, 10, 50)).astype(np.float32)
    tk = np.random.default_rng(1).integers(0, 50, (2, 10))
    s = llm.compare_logits(lg, lg, tk)
    assert s["kl"] == pytest.approx(0, abs=1e-9) and s["top1_agreement"] == 1.0 and s["ppl_delta"] == 0

def test_quantize_rows_streaming_matches_direct():
    w = np.random.default_rng(0).standard_normal((50, 70)).astype(np.float32)
    qt = _quantize_rows_streaming(lambda r: w[r], w.shape, 8, 32, chunk_rows=16)
    direct = qr.quantize_tensor(w, QuantConfig(bits=8, group_size=32))
    np.testing.assert_array_equal(qt.dequantize(), direct.dequantize())
    np.testing.assert_array_equal(qt.dequantize_rows(np.array([3, 49])), direct.dequantize()[[3, 49]])

def test_tokenize_files_windows(tmp_path):
    class Tok:
        def encode(self, text):
            class R:
                ids = list(range(len(text)))

            return R()

    f1, f2 = tmp_path / "a.txt", tmp_path / "b.txt"
    f1.write_text("x" * 100)
    f2.write_text("y" * 100)
    out = llm.tokenize_files(Tok(), [str(f1), str(f2)], 10, 6)
    assert out.shape == (6, 10)
    with pytest.raises(ConfigError):
        llm.tokenize_files(Tok(), [str(f1)], 10, 50)

def test_llm_cli_end_to_end(ckpt_dir, tmp_path):
    import subprocess
    import sys

    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers import models, pre_tokenizers

    vocab = {f"w{i}": i for i in range(V - 1)} | {"<unk>": V - 1}
    tok = tokenizers.Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(os.path.join(ckpt_dir, "tokenizer.json"))
    rng = np.random.default_rng(0)
    for name in ("calib", "heldout"):
        (tmp_path / f"{name}.txt").write_text(" ".join(f"w{i}" for i in rng.integers(0, V - 1, 2000)))

    def run(*args):
        r = subprocess.run([sys.executable, "-m", "quantara", "llm", *args], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        return r.stdout

    assert "qwen2" in run("info", ckpt_dir)
    out = str(tmp_path / "q")
    run(
        "quantize",
        ckpt_dir,
        out,
        "--method",
        "rtn",
        "--bits",
        "8",
        "--group-size",
        "32",
        "--calib-files",
        str(tmp_path / "calib.txt"),
        "--calib-seqs",
        "8",
        "--seq-len",
        "32",
    )
    assert "Quantara output" in run("info", out)
    res = str(tmp_path / "r.json")
    stdout = run(
        "eval",
        out,
        "--reference",
        ckpt_dir,
        "--files",
        str(tmp_path / "heldout.txt"),
        "--seqs",
        "4",
        "--seq-len",
        "32",
        "--json",
        res,
    )
    r = json.load(open(res))
    assert r["top1_agreement"] > 0.9 and "perplexity" in stdout

def test_calibration_on_float_activations(ckpt_dir, tmp_path):
    cfg = llm.LLMQuantConfig(
        ptq=qr.PTQConfig(weight=QuantConfig(bits=8, group_size=32), method="gptq", cache_weights=False),
        calib_from_quantized=False,
    )
    m = llm.quantize_llm(ckpt_dir, str(tmp_path / "q"), cfg, tokens(8, 24, seed=5), max_layers=1)
    assert m["layers"][0]["hidden_cosine"] > 0.99

def test_safetensors_rejects_bad_offsets(tmp_path):
    def write(header):
        h = json.dumps(header).encode()
        p = tmp_path / "x.safetensors"
        p.write_bytes(struct.pack("<Q", len(h)) + h + b"x" * 16)
        return str(p)

    for header in (
        {"w": {"dtype": "F32", "shape": [1], "data_offsets": [-8, 4]}},
        {"w": {"dtype": "F32", "shape": [100], "data_offsets": [0, 16]}},
        {"w": {"dtype": "F32", "shape": [4], "data_offsets": [8, 0]}},
    ):
        with pytest.raises(CheckpointError):
            SafetensorsFile(write(header))
