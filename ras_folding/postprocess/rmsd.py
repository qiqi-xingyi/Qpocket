# Author: Yuqi Zhang
"""CA-only RMSD helpers used by dedup / clustering / basin summary.

No Kabsch alignment — the encoder anchor frame already aligns every
trace at coords[0] (anchor_left), and SQD refinement preserves that
alignment. Adding a Kabsch step here would erase the anchor frame.
"""
from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np


def ca_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Unaligned per-residue CA RMSD between two equally-shaped traces."""
    a = np.asarray(coords_a, dtype=np.float64)
    b = np.asarray(coords_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"expected (n, 3); got {a.shape}")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("non-finite coordinates")
    if a.shape[0] == 0:
        return 0.0
    diffs = a - b
    msd = float(np.mean(np.sum(diffs * diffs, axis=1)))
    return float(math.sqrt(max(msd, 0.0)))


def pairwise_ca_rmsd(candidates: Sequence) -> np.ndarray:
    """(N, N) RMSD matrix over a list of objects exposing
    ``.sample.coords`` (RefinedCandidate-like). Returns zero (0,0) if
    `candidates` is empty.
    """
    n = len(candidates)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    coords_list: List[np.ndarray] = []
    for k, c in enumerate(candidates):
        coords = c.sample.coords if hasattr(c, "sample") else c.coords
        if coords is None:
            raise ValueError(f"candidate {k} has coords=None")
        coords = np.asarray(coords, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                f"candidate {k} coords shape {coords.shape} not (L, 3)"
            )
        if not np.all(np.isfinite(coords)):
            raise ValueError(f"candidate {k} has non-finite coords")
        coords_list.append(coords)
    L = coords_list[0].shape[0]
    for k, c in enumerate(coords_list):
        if c.shape[0] != L:
            raise ValueError(
                f"candidate {k} has length {c.shape[0]} != reference {L}"
            )
    stack = np.stack(coords_list, axis=0)             # (N, L, 3)
    diffs = stack[:, None, :, :] - stack[None, :, :, :]  # (N, N, L, 3)
    msd = np.mean(np.sum(diffs * diffs, axis=-1), axis=-1)  # (N, N)
    np.clip(msd, 0.0, None, out=msd)
    rmsd = np.sqrt(msd)
    # diagonal exactly zero (defensive)
    np.fill_diagonal(rmsd, 0.0)
    return rmsd


__all__ = ["ca_rmsd", "pairwise_ca_rmsd"]
