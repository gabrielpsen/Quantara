import numpy as np
import pytest

from quantara import nn
from quantara.training import cross_entropy

def numgrad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = x[i]
        x[i] = old + eps
        a = f()
        x[i] = old - eps
        b = f()
        x[i] = old
        g[i] = (a - b) / (2 * eps)
    return g

def check(layer, x, seed=0):
    layer.astype(np.float64).train()
    rng = np.random.default_rng(seed)
    out = layer(x)
    probe = rng.standard_normal(out.shape)

    def loss():
        return float(np.sum(layer(x) * probe))

    layer.zero_grad()
    layer(x)
    dx = layer.backward(probe)
    if dx is not None:
        np.testing.assert_allclose(dx, numgrad(loss, x), rtol=1e-4, atol=1e-6)
    for name, p in layer.named_parameters():
        np.testing.assert_allclose(p.grad, numgrad(loss, p.data), rtol=1e-4, atol=1e-6, err_msg=name)

def test_linear():
    check(nn.Linear(5, 4), np.random.default_rng(0).standard_normal((3, 5)))

def test_linear_3d():
    check(nn.Linear(5, 4), np.random.default_rng(0).standard_normal((2, 3, 5)))

@pytest.mark.parametrize("act", [nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid, nn.SiLU])
def test_activations(act):
    x = np.random.default_rng(1).standard_normal((4, 6)) + 0.05
    check(act(), x)

def test_layernorm():
    check(nn.LayerNorm(6), np.random.default_rng(2).standard_normal((3, 4, 6)))

@pytest.mark.parametrize("stride,pad", [(1, 0), (2, 1), (1, 1)])
def test_conv(stride, pad):
    check(
        nn.Conv2d(2, 3, 3, stride=stride, padding=pad), np.random.default_rng(3).standard_normal((2, 2, 7, 7))
    )

def test_sequential_residual():
    net = nn.Sequential(
        [
            nn.Linear(4, 6),
            nn.Tanh(),
            nn.Residual(nn.Sequential([nn.Linear(6, 6), nn.GELU()])),
            nn.Linear(6, 3),
        ]
    )
    check(net, np.random.default_rng(4).standard_normal((5, 4)))

def test_embedding_grad():
    emb = nn.Embedding(7, 3).astype(np.float64).train()
    idx = np.array([[1, 2, 1], [6, 0, 2]])
    out = emb(idx)
    emb.backward(np.ones_like(out))
    assert emb.weight.grad[1].sum() == pytest.approx(2 * 3)
    assert emb.weight.grad[3].sum() == 0

def test_cross_entropy_grad():
    rng = np.random.default_rng(5)
    logits, y = rng.standard_normal((6, 4)), rng.integers(0, 4, 6)
    _, g = cross_entropy(logits, y)
    num = numgrad(lambda: cross_entropy(logits, y)[0], logits)
    np.testing.assert_allclose(g, num, rtol=1e-4, atol=1e-7)

def test_attention_causality():
    att = nn.MultiHeadAttention(16, 4).eval()
    x = np.random.default_rng(6).standard_normal((1, 5, 16)).astype(np.float32)
    y1 = att(x)
    x2 = x.copy()
    x2[0, 4] += 5.0
    y2 = att(x2)
    np.testing.assert_allclose(y1[0, :4], y2[0, :4], atol=1e-5)

def test_spec_roundtrip_and_hooks():
    net = nn.Sequential([nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)])
    seen = []
    remove = net.get_submodule("0").register_forward_hook(lambda m, i, o: seen.append(o.shape))
    net(np.ones((2, 3), np.float32))
    remove()
    net(np.ones((2, 3), np.float32))
    assert seen == [(2, 4)]
    clone = nn.build_from_spec(net.spec())
    clone.load_state_dict(net.state_dict())
    x = np.random.default_rng(0).standard_normal((2, 3)).astype(np.float32)
    np.testing.assert_allclose(clone(x), net(x))
