from __future__ import annotations

import os
from typing import List, Sequence

import numpy as np

from quantara.errors import CheckpointError, ConfigError

def load_tokenizer(model_dir: str):
    try:
        from tokenizers import Tokenizer
    except ImportError as e:
        raise CheckpointError("the 'tokenizers' package is required: pip install tokenizers") from e
    path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(path):
        raise CheckpointError(f"{path} not found")
    return Tokenizer.from_file(path)

def tokenize_files(tokenizer, files: Sequence[str], seq_len: int, n_seqs: int) -> np.ndarray:
    windows: List[np.ndarray] = []
    for f in files:
        with open(f, encoding="utf-8", errors="ignore") as fh:
            ids = tokenizer.encode(fh.read()).ids
        for a in range(0, len(ids) - seq_len + 1, seq_len):
            windows.append(np.asarray(ids[a : a + seq_len], dtype=np.int64))
    if len(windows) < n_seqs:
        raise ConfigError(
            f"only {len(windows)} windows of {seq_len} tokens available; need {n_seqs} - provide more text"
        )
    pick = np.linspace(0, len(windows) - 1, n_seqs).round().astype(int)
    return np.stack([windows[i] for i in pick])
