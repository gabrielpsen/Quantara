from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union

from quantara.core.formats import QuantConfig
from quantara.core.observers import ActConfig

METHODS = ("rtn", "gptq", "awq", "smoothquant")

@dataclass
class PTQConfig:
    weight: QuantConfig = field(default_factory=lambda: QuantConfig(bits=4, group_size=128))
    method: str = "rtn"
    act: Optional[ActConfig] = None
    bit_map: Dict[str, Union[int, QuantConfig]] = field(default_factory=dict)
    skip: Sequence[str] = ()
    quantize_embeddings: bool = True
    min_numel: int = 0
    sequential: bool = False
    gptq_blocksize: int = 128
    gptq_percdamp: float = 0.01
    gptq_act_order: bool = False
    awq_grid: int = 20
    smooth_alpha: float = 0.5
    cache_weights: bool = True
    calib_samples: int = 2048

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}; choose from {METHODS}")

    def cfg_for(self, path: str) -> QuantConfig:
        for pat, v in self.bit_map.items():
            if fnmatch.fnmatch(path, pat):
                return v if isinstance(v, QuantConfig) else self.weight.with_(bits=int(v), format="int")
        return self.weight
