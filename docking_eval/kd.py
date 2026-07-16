# Author: Yuqi Zhang
"""Vina affinity (kcal/mol) → estimated dissociation constant Kd (M).

Uses Kd = exp(ΔG / RT) with R in kcal/(mol·K).

ΔG is typically negative (favorable binding) so Kd < 1 M. We do NOT
report this as an experimentally-validated Kd — variable name is
``estimated_kd_m`` everywhere downstream to avoid confusion.
"""
from __future__ import annotations

import math


_R_KCAL_PER_MOL_K = 0.0019872041


def affinity_to_kd_m(
    delta_g_kcal_mol: float,
    temperature_k: float = 298.15,
) -> float:
    if delta_g_kcal_mol is None:
        raise ValueError("delta_g_kcal_mol is None")
    g = float(delta_g_kcal_mol)
    if math.isnan(g) or math.isinf(g):
        raise ValueError(f"non-finite delta_g: {delta_g_kcal_mol!r}")
    if temperature_k <= 0:
        raise ValueError(f"temperature_k must be > 0; got {temperature_k}")
    rt = _R_KCAL_PER_MOL_K * float(temperature_k)
    return float(math.exp(g / rt))


__all__ = ["affinity_to_kd_m"]
