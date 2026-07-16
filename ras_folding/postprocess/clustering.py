# Author: Yuqi Zhang
"""Connected-component basin clustering on the CA-RMSD graph."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ras_folding.postprocess.rmsd import pairwise_ca_rmsd


def cluster_basins_by_rmsd(
    candidates: Sequence,
    threshold: float,
) -> Tuple[Dict[int, int], Dict[str, Any]]:
    """Connected components in the graph G = (V, E) where:
        V = candidate indices
        E = {(i, j) | RMSD(i, j) < threshold, i != j}

    Returns
    -------
    cluster_assignments : dict[candidate_index → basin_id], basin_id 0-based.
    summary : dict with n_basins, basin_rmsd_threshold, basin_sizes,
              mean_pairwise_rmsd_global.
    """
    n = len(candidates)
    if n == 0:
        return {}, {
            "n_basins": 0,
            "basin_rmsd_threshold": float(threshold),
            "basin_sizes": [],
            "mean_pairwise_rmsd_global": None,
        }
    if n == 1:
        return {0: 0}, {
            "n_basins": 1,
            "basin_rmsd_threshold": float(threshold),
            "basin_sizes": [1],
            "mean_pairwise_rmsd_global": 0.0,
        }

    rmsd = pairwise_ca_rmsd(candidates)

    # Union-Find
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    iu, ju = np.triu_indices(n, k=1)
    edges_mask = rmsd[iu, ju] < float(threshold)
    for i, j, has_edge in zip(iu.tolist(), ju.tolist(), edges_mask.tolist()):
        if has_edge:
            union(i, j)

    # Map roots → basin_id (0-based, in order of first appearance)
    roots: Dict[int, int] = {}
    assignments: Dict[int, int] = {}
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        assignments[i] = roots[r]

    # basin_sizes (ordered by basin_id)
    n_basins = len(roots)
    sizes = [0] * n_basins
    for bid in assignments.values():
        sizes[bid] += 1

    # Global mean pairwise RMSD (upper triangle, i < j)
    if n >= 2:
        mean_global = float(rmsd[iu, ju].mean())
    else:
        mean_global = 0.0

    summary: Dict[str, Any] = {
        "n_basins": int(n_basins),
        "basin_rmsd_threshold": float(threshold),
        "basin_sizes": [int(s) for s in sizes],
        "mean_pairwise_rmsd_global": mean_global,
    }
    return assignments, summary


__all__ = ["cluster_basins_by_rmsd"]
