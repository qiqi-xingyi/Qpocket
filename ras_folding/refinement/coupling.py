# Author: Yuqi Zhang
"""Off-diagonal coupling helpers for the subspace effective Hamiltonian.

Implemented coupling: rmsd_kernel
    T_ij = -kappa * exp(-RMSD_ij^2 / (2 * sigma^2))
restricted to the K nearest neighbours of each i in RMSD space.

The pairwise RMSD here is the unaligned (no Kabsch) per-residue Euclidean
RMSD between coordinate arrays of the same length:

    RMSD(A, B) = sqrt( mean_i ||A_i - B_i||^2 )

This is appropriate because every candidate in the subspace is anchored
(``ca[0] == anchor_left``) and shares the same anchor_right tolerance,
so the alignment is fixed by the encoder's anchoring convention. Using
unaligned RMSD also keeps the coupling deterministic and cheap.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------- #
# pairwise RMSD                                                          #
# ---------------------------------------------------------------------- #

def pairwise_rmsd_matrix(coords_stack: np.ndarray) -> np.ndarray:
    """Return (N, N) RMSD matrix for `coords_stack` of shape (N, L, 3).

    Definition: RMSD_ij = sqrt( mean over residues of squared CA-CA
    displacement between coords_stack[i] and coords_stack[j]).
    """
    if coords_stack.ndim != 3 or coords_stack.shape[-1] != 3:
        raise ValueError(
            f"expected coords_stack shape (N, L, 3), got {coords_stack.shape}"
        )
    N, L, _ = coords_stack.shape
    if N == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if L == 0:
        return np.zeros((N, N), dtype=np.float64)
    # broadcasted differences: (N, 1, L, 3) - (1, N, L, 3) → (N, N, L, 3)
    diffs = coords_stack[:, None, :, :] - coords_stack[None, :, :, :]
    sq = np.sum(diffs * diffs, axis=-1)            # (N, N, L)
    msd = sq.mean(axis=-1)                          # (N, N)
    # numerical safety: clip tiny negatives that may arise from FP slack
    np.clip(msd, 0.0, None, out=msd)
    return np.sqrt(msd)


# ---------------------------------------------------------------------- #
# rmsd kernel coupling                                                   #
# ---------------------------------------------------------------------- #

def _knn_indices(rmsd: np.ndarray, k: int) -> np.ndarray:
    """For each row of rmsd, return indices of the k smallest entries
    (excluding self). Returns shape (N, k_eff) where k_eff = min(k, N-1)."""
    N = rmsd.shape[0]
    if N <= 1:
        return np.zeros((N, 0), dtype=np.int64)
    k_eff = max(0, min(int(k), N - 1))
    if k_eff == 0:
        return np.zeros((N, 0), dtype=np.int64)
    rmsd_no_self = rmsd.copy()
    np.fill_diagonal(rmsd_no_self, np.inf)
    # argpartition is faster than full sort
    part = np.argpartition(rmsd_no_self, kth=k_eff - 1, axis=1)[:, :k_eff]
    return part.astype(np.int64, copy=False)


def rmsd_kernel_coupling(
    coords_stack: np.ndarray,
    *,
    k_neighbors: int = 20,
    kappa: float = 0.2,
    sigma: float = None,
    eps: float = 1e-9,
) -> Tuple[sp.csr_matrix, dict]:
    """Build the symmetric KNN-RMSD coupling matrix T (sparse).

    Returns
    -------
    T : scipy.sparse.csr_matrix of shape (N, N), symmetric. Entries:
        T[i, j] = -kappa * exp(-RMSD_ij^2 / (2 sigma^2)) iff
        j is in KNN(i) OR i is in KNN(j); 0 otherwise. Diagonal is 0.
    info : dict — sigma_used, n_nonzero, coupling_min, coupling_max
    """
    if kappa < 0:
        raise ValueError(f"kappa must be >= 0, got {kappa}")
    rmsd = pairwise_rmsd_matrix(coords_stack)
    N = rmsd.shape[0]
    info = {"N": N}

    if N <= 1:
        T = sp.csr_matrix((N, N), dtype=np.float64)
        info["sigma_used"] = float("nan")
        info["n_nonzero"] = 0
        info["coupling_min"] = 0.0
        info["coupling_max"] = 0.0
        return T, info

    knn = _knn_indices(rmsd, k_neighbors)

    # sigma default: median of KNN RMSD distances
    if sigma is None:
        rows = np.repeat(np.arange(N), knn.shape[1])
        cols = knn.reshape(-1)
        if cols.size:
            sigma_val = float(np.median(rmsd[rows, cols]))
        else:
            sigma_val = 0.0
        if sigma_val < eps:
            sigma_val = eps
    else:
        sigma_val = float(sigma)
        if sigma_val < eps:
            sigma_val = eps
    info["sigma_used"] = sigma_val

    # Build sparse symmetric T from KNN union.
    rows = np.repeat(np.arange(N), knn.shape[1])
    cols = knn.reshape(-1)
    # symmetrize: include (i, j) and (j, i)
    all_rows = np.concatenate([rows, cols])
    all_cols = np.concatenate([cols, rows])
    pair_dists = rmsd[all_rows, all_cols]
    weights = -kappa * np.exp(
        -(pair_dists * pair_dists) / (2.0 * sigma_val * sigma_val)
    )
    # remove self-loops
    self_mask = (all_rows != all_cols)
    all_rows = all_rows[self_mask]
    all_cols = all_cols[self_mask]
    weights = weights[self_mask]

    # build COO; duplicate (i,j) entries from the symmetrization will be
    # summed by csr_matrix; deduplicate by averaging via division by 2 of
    # the duplicates afterwards.
    T = sp.csr_matrix(
        (weights, (all_rows, all_cols)), shape=(N, N), dtype=np.float64,
    )
    # csr_matrix with duplicate (i,j) entries will sum them; we want to
    # keep the value (since (i,j) and (j,i) appear once each in our
    # construction the sum is 2*w; halve it.)
    T = T.multiply(0.5)
    # ensure symmetry exactly
    T = (T + T.T).multiply(0.5)
    # remove tiny numerical noise on diagonal
    T = T.tolil()
    T.setdiag(0.0)
    T = T.tocsr()
    T.eliminate_zeros()

    info["n_nonzero"] = int(T.nnz)
    if T.nnz > 0:
        data = T.data
        info["coupling_min"] = float(data.min())
        info["coupling_max"] = float(data.max())
    else:
        info["coupling_min"] = 0.0
        info["coupling_max"] = 0.0
    info["k_neighbors_effective"] = int(knn.shape[1])
    info["kappa"] = float(kappa)
    return T, info


__all__ = [
    "pairwise_rmsd_matrix",
    "rmsd_kernel_coupling",
]
