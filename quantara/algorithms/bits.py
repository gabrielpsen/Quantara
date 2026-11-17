from __future__ import annotations

from typing import Dict

def allocate_bits(
    sensitivity: Dict[str, Dict[int, float]], numel: Dict[str, int], target_avg_bits: float
) -> Dict[str, int]:
    names = list(sensitivity)
    options = {n: sorted(sensitivity[n]) for n in names}
    cur = {n: options[n][0] for n in names}
    total = float(sum(numel[n] for n in names))
    budget = target_avg_bits * total
    used = sum(cur[n] * numel[n] for n in names)
    while True:
        best, best_ratio = None, 0.0
        for n in names:
            opts = options[n]
            i = opts.index(cur[n])
            if i + 1 >= len(opts):
                continue
            nxt = opts[i + 1]
            cost = (nxt - cur[n]) * numel[n]
            if used + cost > budget + 1e-9:
                continue
            gain = sensitivity[n][cur[n]] - sensitivity[n][nxt]
            ratio = gain / cost
            if ratio > best_ratio:
                best, best_ratio = (n, nxt, cost), ratio
        if best is None:
            break
        n, nxt, cost = best
        cur[n] = nxt
        used += cost
    return cur
