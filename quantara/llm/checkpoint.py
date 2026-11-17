from __future__ import annotations

import json
import math
import mmap
import os
import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from quantara.errors import CheckpointError

_DTYPES = {
    "F32": np.float32,
    "F16": np.float16,
    "F64": np.float64,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}

_ITEMSIZE = {**{k: np.dtype(v).itemsize for k, v in _DTYPES.items()}, "BF16": 2}

@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    start: int
    end: int

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def _bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    # bf16 is the top half of a float32.
    return (raw.astype(np.uint32) << 16).view(np.float32)

class SafetensorsFile:
    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "rb")
        try:
            size = os.fstat(self._fh.fileno()).st_size
            if size < 8:
                raise CheckpointError(f"{path}: file too small to be a safetensors file")
            (hlen,) = struct.unpack("<Q", self._fh.read(8))
            if hlen > size - 8 or hlen > 100 * 1024 * 1024:
                raise CheckpointError(f"{path}: invalid safetensors header length")
            header = json.loads(self._fh.read(hlen))
            self._base = 8 + hlen
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._fh.close()
            raise
        header.pop("__metadata__", None)
        self.tensors: Dict[str, TensorInfo] = {}
        for name, h in header.items():
            s, e = h["data_offsets"]
            info = TensorInfo(name, h["dtype"], tuple(h["shape"]), s, e)
            if not 0 <= s <= e or self._base + e > size:
                raise CheckpointError(f"{path}: tensor {name!r} has invalid offsets (truncated download?)")
            if info.dtype in _ITEMSIZE and e - s != info.numel * _ITEMSIZE[info.dtype]:
                raise CheckpointError(f"{path}: tensor {name!r} shape does not match its byte range")
            self.tensors[name] = info

    def _raw(self, info: TensorInfo) -> np.ndarray:
        if info.dtype == "BF16":
            dt = np.uint16
        elif info.dtype in _DTYPES:
            dt = _DTYPES[info.dtype]
        else:
            raise CheckpointError(f"unsupported dtype {info.dtype} for tensor {info.name!r}")
        return np.frombuffer(self._mm, dtype=dt, count=info.numel, offset=self._base + info.start)

    def get(self, name: str, dtype=np.float32) -> np.ndarray:
        info = self.tensors[name]
        raw = self._raw(info)
        out = _bf16_to_f32(raw) if info.dtype == "BF16" else raw
        if np.issubdtype(out.dtype, np.floating):
            out = out.astype(dtype, copy=False)
        return np.array(out).reshape(info.shape)

    def get_rows(self, name: str, rows: np.ndarray, dtype=np.float32) -> np.ndarray:
        info = self.tensors[name]
        if len(info.shape) != 2:
            raise CheckpointError("get_rows needs a 2-D tensor")
        raw = self._raw(info).reshape(info.shape)
        sel = raw[np.asarray(rows)]
        out = _bf16_to_f32(sel) if info.dtype == "BF16" else sel
        return out.astype(dtype, copy=False)

    def close(self) -> None:
        try:
            self._mm.close()
        finally:
            self._fh.close()

class Checkpoint:
    def __init__(self, path: str):
        if not os.path.isdir(path):
            raise CheckpointError(f"{path} is not a directory")
        self.path = path
        cfg_path = os.path.join(path, "config.json")
        self.config: dict = _read_json(cfg_path) if os.path.exists(cfg_path) else {}
        index = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index):
            weight_map = _read_json(index)["weight_map"]
            shards = sorted(set(weight_map.values()))
        else:
            shards = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
        if not shards:
            raise CheckpointError(f"no .safetensors files found in {path}")
        self._files: Dict[str, SafetensorsFile] = {}
        self._where: Dict[str, str] = {}
        for s in shards:
            f = SafetensorsFile(os.path.join(path, s))
            self._files[s] = f
            for n in f.tensors:
                self._where[n] = s

    def names(self) -> List[str]:
        return list(self._where)

    def __contains__(self, name: str) -> bool:
        return name in self._where

    def info(self, name: str) -> TensorInfo:
        return self._file(name).tensors[name]

    def _file(self, name: str) -> SafetensorsFile:
        if name not in self._where:
            raise CheckpointError(f"tensor {name!r} not found in checkpoint")
        return self._files[self._where[name]]

    def get(self, name: str, dtype=np.float32) -> np.ndarray:
        return self._file(name).get(name, dtype)

    def get_rows(self, name: str, rows, dtype=np.float32) -> np.ndarray:
        return self._file(name).get_rows(name, rows, dtype)

    def total_bytes(self) -> int:
        return sum(i.end - i.start for f in self._files.values() for i in f.tensors.values())

    def num_parameters(self) -> int:
        return sum(i.numel for f in self._files.values() for i in f.tensors.values())

    def close(self) -> None:
        for f in self._files.values():
            f.close()

    def __enter__(self) -> Checkpoint:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

def write_safetensors(tensors: Dict[str, np.ndarray], path: str) -> None:
    rev = {np.dtype(v): k for k, v in _DTYPES.items()}
    header, offset, blobs = {}, 0, []
    for name, t in tensors.items():
        t = np.ascontiguousarray(t)
        header[name] = {
            "dtype": rev[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + t.nbytes],
        }
        blobs.append(t.tobytes())
        offset += t.nbytes
    h = json.dumps(header).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)
