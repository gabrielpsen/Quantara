from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class LayerReport:
    name: str
    kind: str
    shape: Tuple[int, ...]
    config: str
    method: str
    bits_per_weight: float
    original_bytes: int
    quantized_bytes: int
    weight_rel_error: float
    output_mse: Optional[float] = None

@dataclass
class QuantReport:
    layers: List[LayerReport] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    original_bytes: int = 0
    quantized_bytes: int = 0

    @property
    def compression(self) -> float:
        return self.original_bytes / max(self.quantized_bytes, 1)

    def table(self) -> str:
        head = f"{'layer':<28}{'config':<22}{'method':<12}{'bpw':>6}{'w-err':>10}{'out-mse':>12}"
        rows = [head, "-" * len(head)]
        for r in self.layers:
            om = "-" if r.output_mse is None else f"{r.output_mse:.3e}"
            rows.append(
                f"{r.name:<28}{r.config:<22}{r.method:<12}{r.bits_per_weight:>6.2f}"
                f"{r.weight_rel_error:>10.4f}{om:>12}"
            )
        rows.append("-" * len(head))
        rows.append(
            f"model: {self.original_bytes / 1e6:.3f} MB -> {self.quantized_bytes / 1e6:.3f} MB "
            f"({self.compression:.2f}x smaller)"
        )
        if self.skipped:
            rows.append("kept in float: " + ", ".join(self.skipped))
        return "\n".join(rows)
