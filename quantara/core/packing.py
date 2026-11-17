from __future__ import annotations

import numpy as np

_CHUNK = 1 << 22

def packed_nbytes(count: int, bits: int) -> int:
    return (count * bits + 7) // 8

def _check(bits: int) -> None:
    if not 1 <= bits <= 8:
        raise ValueError("bits must be in [1, 8]")

def _pack_generic(flat: np.ndarray, bits: int) -> np.ndarray:
    out = []
    for i in range(0, flat.size, _CHUNK):
        part = flat[i : i + _CHUNK]
        b = np.unpackbits(part[:, None], axis=1)[:, 8 - bits :]
        out.append(np.packbits(b.reshape(-1)))
    return np.concatenate(out) if out else np.empty(0, dtype=np.uint8)

# element 0 sits in the high bits of byte 0 for every width.
def pack(codes: np.ndarray, bits: int) -> np.ndarray:
    _check(bits)
    flat = np.ascontiguousarray(codes, dtype=np.uint8).reshape(-1)
    if flat.size and bits < 8 and int(flat.max()) >= (1 << bits):
        raise ValueError(f"code out of range for {bits} bits")
    if bits == 8:
        return flat.copy()
    n = flat.size
    if bits == 4:
        if n % 2:
            flat = np.concatenate([flat, np.zeros(1, np.uint8)])
        return (flat[0::2] << 4) | flat[1::2]
    if bits == 2:
        pad = (-n) % 4
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, np.uint8)])
        return (flat[0::4] << 6) | (flat[1::4] << 4) | (flat[2::4] << 2) | flat[3::4]
    if bits == 1:
        return np.packbits(flat)
    return _pack_generic(flat, bits)

def unpack(packed: np.ndarray, bits: int, count: int) -> np.ndarray:
    _check(bits)
    p = np.ascontiguousarray(packed, dtype=np.uint8)
    if p.size < packed_nbytes(count, bits):
        raise ValueError("packed buffer is too small for the requested element count")
    if bits == 8:
        return p[:count].copy()
    if bits == 4:
        out = np.empty(p.size * 2, dtype=np.uint8)
        out[0::2] = p >> 4
        out[1::2] = p & 0x0F
        return out[:count]
    if bits == 2:
        out = np.empty(p.size * 4, dtype=np.uint8)
        out[0::4] = p >> 6
        out[1::4] = (p >> 4) & 3
        out[2::4] = (p >> 2) & 3
        out[3::4] = p & 3
        return out[:count]
    if bits == 1:
        return np.unpackbits(p)[:count]
    out = np.empty(count, dtype=np.uint8)
    step_bytes = _CHUNK * bits // 8
    for i in range(0, count, _CHUNK):
        n = min(_CHUNK, count - i)
        seg = p[(i * bits) // 8 : (i * bits) // 8 + step_bytes + 1]
        b = np.unpackbits(seg)[: n * bits].reshape(n, bits)
        full = np.zeros((n, 8), dtype=np.uint8)
        full[:, 8 - bits :] = b
        out[i : i + n] = np.packbits(full, axis=1).reshape(-1)
    return out
