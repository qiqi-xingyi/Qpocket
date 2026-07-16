# Author: Yuqi Zhang
"""Assemble the SQD-inspired effective Hamiltonian H_eff.

H_eff[i, j] = E_norm(z_i) * delta_ij + T(z_i, z_j)

where:
  - E_norm comes from `energy_normalization` ("mad" or "zscore")
  - T(z_i, z_j) is the KNN-RMSD coupling (see refinement.coupling)
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import scipy.sparse as sp


def normalize_energies(
    energies: np.ndarray,
    method: str = "mad",
    eps: float = 1e-9,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Normalize an energy vector. Returns (E_norm, info).

    method = "mad":
        median_E = median(E)
        mad_E    = median(|E - median_E|)
        E_norm   = (E - median_E) / (mad_E + eps)

    method = "zscore":
        E_norm = (E - mean(E)) / (std(E) + eps)

    Fallback chain:
      mad -> if mad < eps -> zscore -> if std < eps -> all zeros
    """
    E = np.asarray(energies, dtype=np.float64)
    info: Dict[str, Any] = {"method_requested": method}
    if E.size == 0:
        info["fallback"] = "empty"
        return np.zeros_like(E), info

    if method == "mad":
        med = float(np.median(E))
        mad = float(np.median(np.abs(E - med)))
        info["median"] = med
        info["mad"] = mad
        if mad < eps:
            # fallback to zscore
            std = float(np.std(E))
            info["fallback"] = "mad_too_small_use_zscore"
            info["std"] = std
            if std < eps:
                info["fallback_2"] = "std_too_small_use_zeros"
                return np.zeros_like(E), info
            return (E - float(np.mean(E))) / (std + eps), info
        return (E - med) / (mad + eps), info

    if method == "zscore":
        mu = float(np.mean(E))
        std = float(np.std(E))
        info["mean"] = mu
        info["std"] = std
        if std < eps:
            info["fallback"] = "std_too_small_use_zeros"
            return np.zeros_like(E), info
        return (E - mu) / (std + eps), info

    raise ValueError(f"unsupported energy_normalization method: {method!r}")


def build_effective_hamiltonian(
    energies: np.ndarray,
    coupling_matrix,                # sparse or dense (N, N)
    energy_normalization: str = "mad",
    eps: float = 1e-9,
) -> Tuple[sp.csr_matrix, np.ndarray, Dict[str, Any]]:
    """Compose H_eff from a diagonal of normalized energies + a coupling.

    Returns (H_eff_sparse, E_norm, info). The diagonal is normalized;
    the off-diagonal block is the supplied coupling. Result is forced
    symmetric.
    """
    E_norm, norm_info = normalize_energies(
        energies, method=energy_normalization, eps=eps,
    )

    N = E_norm.shape[0]
    if sp.issparse(coupling_matrix):
        H_off = coupling_matrix.tocsr()
    else:
        H_off = sp.csr_matrix(coupling_matrix)
    if H_off.shape != (N, N):
        raise ValueError(
            f"coupling shape {H_off.shape} != (N, N) = ({N}, {N})"
        )
    # force diag of off to zero
    H_off = H_off.tolil()
    H_off.setdiag(0.0)
    H_off = H_off.tocsr()

    diag = sp.diags(E_norm, format="csr")
    H = (diag + H_off).tocsr()
    # Symmetrize just to absorb FP noise (coupling builder already
    # symmetrizes; a second pass is cheap and protects callers that pass
    # arbitrary couplings).
    H = ((H + H.T).multiply(0.5)).tocsr()

    info: Dict[str, Any] = {
        "energy_normalization": norm_info,
        "n_nonzero_off_diagonal": int(
            H_off.nnz - np.count_nonzero(H_off.diagonal())
        ),
        "n_nonzero_total": int(H.nnz),
    }
    return H, E_norm, info


__all__ = ["normalize_energies", "build_effective_hamiltonian"]
