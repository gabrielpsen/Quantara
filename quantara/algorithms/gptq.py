from __future__ import annotations

import numpy as np

from quantara.core.formats import fit_params, get_quantizer, QuantConfig
from quantara.core.tensor import QTensor

def gptq(
    w: np.ndarray,
    H: np.ndarray,
    cfg: QuantConfig,
    blocksize: int = 128,
    percdamp: float = 0.01,
    act_order: bool = False,
    dtype=None,
) -> QTensor:
    if cfg.granularity == "tensor":
        raise ValueError("gptq supports channel or group granularity, not tensor")
    shape = tuple(w.shape)
    W = np.asarray(w).reshape(shape[0], -1)
    R, C = W.shape
    # wide layers use float32 so their Hessian fits in memory.
    dtype = dtype or (np.float64 if C <= 8192 else np.float32)
    W = W.astype(dtype)
    if H.shape != (C, C):
        raise ValueError(f"Hessian shape {H.shape} does not match weight columns {C}")
    qz = get_quantizer(cfg)
    L = C if cfg.granularity == "channel" else min(cfg.group_size, C)
    pad = (-C) % L
    Cp = C + pad
    ng = Cp // L

    Hh = np.zeros((Cp, Cp), dtype=dtype)
    Hh[:C, :C] = H
    Wp = np.zeros((R, Cp), dtype=dtype)
    Wp[:, :C] = W
    if pad:
        # padded columns get an identity Hessian so they stay at zero.
        Hh[np.arange(C, Cp), np.arange(C, Cp)] = 1.0
    dead = np.diag(Hh) == 0
    Hh[dead, dead] = 1.0
    Wp[:, dead] = 0.0
    Hh[np.diag_indices(Cp)] += percdamp * np.mean(np.diag(Hh))

    perm = np.arange(Cp)
    if act_order:
        perm[:C] = np.argsort(-np.diag(Hh)[:C], kind="stable")
        Wp = Wp[:, perm]
        Hh = Hh[np.ix_(perm, perm)]
    try:
        Hinv = np.linalg.inv(Hh)
        U = np.linalg.cholesky(Hinv).T
    except np.linalg.LinAlgError:
        Hh[np.diag_indices(Cp)] += 1e-2 * np.mean(np.diag(Hh))
        U = np.linalg.cholesky(np.linalg.inv(Hh)).T

    static = None
    if act_order:
        orig = np.zeros((R, Cp), dtype=dtype)
        orig[:, :C] = W
        orig[:, dead] = 0.0
        sc, zr = fit_params(qz, orig.reshape(R * ng, L).astype(np.float32), cfg)
        static = (sc.reshape(R, ng), None if zr is None else zr.reshape(R, ng))

    scales = np.zeros((R, ng), dtype=np.float32)
    zeros = None
    codes = np.zeros((R, Cp), dtype=np.uint8)
    B = max(L, (blocksize // L) * L) if blocksize > L else L
    cur_scale = cur_zero = None

    def store(gi, sc, zr):
        nonlocal zeros
        scales[:, gi : gi + 1] = sc
        if zr is not None:
            if zeros is None:
                zeros = np.zeros((R, ng), dtype=np.float32)
            zeros[:, gi : gi + 1] = zr

    for i1 in range(0, Cp, B):
        i2 = min(i1 + B, Cp)
        W1 = Wp[:, i1:i2].copy()
        Err1 = np.zeros_like(W1)
        U1 = U[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if static is not None:
                gi = perm[col] // L
                cur_scale = static[0][:, gi : gi + 1]
                cur_zero = None if static[1] is None else static[1][:, gi : gi + 1]
                store(gi, cur_scale, cur_zero)
            elif col % L == 0:
                cur_scale, cur_zero = fit_params(qz, W1[:, i : i + L].astype(np.float32), cfg)
                store(col // L, cur_scale, cur_zero)
            w = W1[:, i : i + 1].astype(np.float32)
            q = qz.encode(w, cur_scale, cur_zero)
            dq = qz.decode(q, cur_scale, cur_zero).astype(dtype)
            codes[:, col : col + 1] = q
            err = (W1[:, i : i + 1] - dq) / U1[i, i]
            W1[:, i:] -= err @ U1[i : i + 1, i:]
            Err1[:, i : i + 1] = err
        Wp[:, i2:] -= Err1 @ U[i1:i2, i2:]

    if act_order:
        inv = np.argsort(perm)
        codes = codes[:, inv]
    flat_codes = codes.reshape(R * ng, L)
    zero_arr = None if zeros is None else zeros.reshape(R * ng, 1)
    return QTensor.create(flat_codes, scales.reshape(R * ng, 1), zero_arr, cfg, shape, pad)
