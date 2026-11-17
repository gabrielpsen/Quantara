from __future__ import annotations

import numpy as np

from quantara.core.formats import QuantConfig
from quantara.core.tensor import QTensor, quantize_tensor

def rtn(w: np.ndarray, cfg: QuantConfig) -> QTensor:
    return quantize_tensor(w, cfg)
