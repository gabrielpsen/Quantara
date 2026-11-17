from __future__ import annotations

import fnmatch
import json
import zipfile
from typing import Dict, Union

import numpy as np

from quantara.core.formats import QuantConfig
from quantara.core.tensor import QTensor, quantize_tensor
from quantara.nn.module import build_from_spec, Module

FORMAT_VERSION = 1

_MAGIC = "quantara"

MAX_MANIFEST_BYTES = 64 << 20

MAX_COMPRESSION_RATIO = 1000

def _write(path: str, manifest: dict, arrays: Dict[str, np.ndarray], compress: bool = False) -> None:
    manifest = {"magic": _MAGIC, "version": FORMAT_VERSION, **manifest}
    blob = np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8)
    with open(path, "wb") as f:
        save = np.savez_compressed if compress else np.savez
        save(f, __manifest__=blob, **{k: np.asarray(v) for k, v in arrays.items()})

def _check_archive(path: str) -> None:
    if not zipfile.is_zipfile(path):
        raise ValueError(f"{path} is not a Quantara archive")
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.filename == "__manifest__.npy" and info.file_size > MAX_MANIFEST_BYTES:
                raise ValueError(f"{path}: manifest is too large")
            ratio = info.file_size / max(info.compress_size, 1)
            if info.file_size > MAX_MANIFEST_BYTES and ratio > MAX_COMPRESSION_RATIO:
                raise ValueError(f"{path}: member {info.filename!r} has a suspicious compression ratio")

def _read(path: str):
    _check_archive(path)
    z = np.load(path, allow_pickle=False)
    manifest = json.loads(bytes(z["__manifest__"]).decode("utf-8"))
    if manifest.get("magic") != _MAGIC:
        raise ValueError(f"{path} is not a Quantara archive")
    if manifest["version"] > FORMAT_VERSION:
        raise ValueError(f"archive version {manifest['version']} is newer than this library")
    return manifest, {k: z[k] for k in z.files if k != "__manifest__"}

def _key(k: str) -> str:
    return k.replace("/", "%2F")

def save_model(model: Module, path: str, compress: bool = False) -> None:
    sd = model.state_dict()
    _write(path, {"kind": "model", "spec": model.spec()}, {_key(k): v for k, v in sd.items()}, compress)

def load_model(path: str) -> Module:
    manifest, arrays = _read(path)
    if manifest["kind"] != "model":
        raise ValueError("archive does not contain a model; use load_tensors()")
    model = build_from_spec(manifest["spec"])
    model.load_state_dict({k.replace("%2F", "/"): v for k, v in arrays.items()})
    return model.eval()

def save_tensors(tensors: Dict[str, Union[QTensor, np.ndarray]], path: str, compress: bool = False) -> None:
    arrays, entries = {}, {}
    for i, (name, t) in enumerate(tensors.items()):
        if isinstance(t, QTensor):
            for k, v in t.arrays().items():
                arrays[f"t{i}.{k}"] = v
            entries[name] = {"id": i, "quantized": True, "meta": t.meta()}
        else:
            arrays[f"t{i}.raw"] = np.asarray(t)
            entries[name] = {"id": i, "quantized": False}
    _write(path, {"kind": "tensors", "tensors": entries}, arrays, compress)

def load_tensors(path: str) -> Dict[str, Union[QTensor, np.ndarray]]:
    manifest, arrays = _read(path)
    if manifest["kind"] != "tensors":
        raise ValueError("archive does not contain loose tensors; use load_model()")
    out = {}
    for name, e in manifest["tensors"].items():
        i = e["id"]
        if e["quantized"]:
            parts = {k.split(".", 1)[1]: v for k, v in arrays.items() if k.startswith(f"t{i}.")}
            out[name] = QTensor.from_parts(parts, e["meta"])
        else:
            out[name] = arrays[f"t{i}.raw"]
    return out

def quantize_state_dict(
    weights: Dict[str, np.ndarray], cfg: QuantConfig, min_numel: int = 1024, skip=()
) -> Dict[str, Union[QTensor, np.ndarray]]:
    out: Dict[str, Union[QTensor, np.ndarray]] = {}
    for name, w in weights.items():
        w = np.asarray(w)
        keep = (
            w.ndim < 2
            or w.size < min_numel
            or any(fnmatch.fnmatch(name, p) for p in skip)
            or not np.issubdtype(w.dtype, np.floating)
        )
        out[name] = w if keep else quantize_tensor(w.astype(np.float32), cfg)
    return out
