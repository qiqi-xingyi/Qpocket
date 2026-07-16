# Author: Yuqi Zhang
"""Hybrid coupling = α · T_Pauli + β · T_RMSD.

Combines:
  - T_Pauli  (math-rigorous): exact matrix elements of Ĥ off-diagonal
              part on the sampled subspace
  - T_RMSD   (biologically-aligned): structural similarity via RMSD
              Gaussian kernel on the KNN graph

The two captures different graphs on the sampled subspace:
  T_Pauli connects bitstrings differing by 1 bit-flip
          (== one of n_qubits encoder bits changes)
  T_RMSD  connects structures that are nearby in 3D
          (== similar CA traces regardless of bit-encoding)

Both are valid notions of "neighbour" in different senses. The hybrid
weighted sum is the SQD-refinement off-diagonal that grounds the
mathematical rigour (Pauli) in biological similarity (RMSD).

Default weights α = β = 0.5. Both can be swept.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from ras_folding.refinement.coupling import rmsd_kernel_coupling
from ras_folding.refinement.pauli_coupling import (
    pauli_hamming1_matrix,
    hamming_graph_stats,
)


def hybrid_coupling_matrix(
    bitstrings: List[int],
    coords_stack: np.ndarray,
    g: float,
    n_qubits: int,
    *,
    alpha_pauli: float = 0.5,
    alpha_rmsd: float = 0.5,
    k_neighbors_rmsd: int = 20,
    kappa_rmsd: float = 0.2,
    sigma_rmsd: Optional[float] = None,
) -> Tuple[sp.csr_matrix, Dict[str, Any]]:
    """Build T = α_pauli · T_Pauli + α_rmsd · T_RMSD.

    Parameters
    ----------
    bitstrings    : list of N integer-encoded bitstrings
    coords_stack  : (N, L, 3) array of decoded CA coordinates
    g             : transverse-field coupling for Pauli part
    n_qubits      : circuit width
    alpha_pauli   : weight on the Pauli (Hamming-1) coupling
    alpha_rmsd    : weight on the RMSD-Gaussian kernel coupling
    k_neighbors_rmsd, kappa_rmsd, sigma_rmsd : RMSD-kernel parameters
                    (forwarded to ``rmsd_kernel_coupling``)

    Returns
    -------
    T_hybrid : N×N sparse matrix
    info     : diagnostics including both components' stats
    """
    if alpha_pauli < 0 or alpha_rmsd < 0:
        raise ValueError("alpha weights must be >= 0")
    if alpha_pauli + alpha_rmsd <= 0:
        raise ValueError("at least one alpha must be > 0")

    N = len(bitstrings)
    if N != coords_stack.shape[0]:
        raise ValueError(
            f"bitstrings length {N} != coords_stack.shape[0] "
            f"{coords_stack.shape[0]}"
        )

    # Pauli part (exact)
    T_pauli = pauli_hamming1_matrix(bitstrings, g=g, n_qubits=n_qubits)
    pauli_stats = hamming_graph_stats(bitstrings, n_qubits=n_qubits)

    # RMSD-kernel part
    T_rmsd, rmsd_info = rmsd_kernel_coupling(
        coords_stack,
        k_neighbors=k_neighbors_rmsd,
        kappa=kappa_rmsd,
        sigma=sigma_rmsd,
    )

    # Weighted sum
    T_hybrid = (alpha_pauli * T_pauli) + (alpha_rmsd * T_rmsd)
    T_hybrid = T_hybrid.tocsr()
    # Force exact symmetry (tiny FP noise from sparse arithmetic)
    T_hybrid = (T_hybrid + T_hybrid.T).multiply(0.5).tocsr()
    T_hybrid.eliminate_zeros()

    info: Dict[str, Any] = {
        "mode": "hybrid",
        "alpha_pauli": float(alpha_pauli),
        "alpha_rmsd": float(alpha_rmsd),
        "g_quantum": float(g),
        "pauli_stats": pauli_stats,
        "rmsd_info": {
            k: (float(v) if isinstance(v, (int, float))
                else v)
            for k, v in rmsd_info.items()
        },
        "n_nonzero_hybrid": int(T_hybrid.nnz),
    }
    return T_hybrid, info


# ---------------------------------------------------------------------- #
# Self-test                                                              #
# ---------------------------------------------------------------------- #

def _self_test() -> None:
    """Check degenerate-weight limits reduce to pure components."""
    rng = np.random.default_rng(42)
    N = 5
    L = 4
    bs = [0, 1, 2, 3, 7]
    coords = rng.normal(size=(N, L, 3))

    # alpha_pauli=1, alpha_rmsd=0 → pure Pauli
    T_pure_pauli, _ = hybrid_coupling_matrix(
        bs, coords, g=0.3, n_qubits=4,
        alpha_pauli=1.0, alpha_rmsd=0.0,
    )
    T_pauli_ref = pauli_hamming1_matrix(bs, g=0.3, n_qubits=4)
    assert np.allclose(T_pure_pauli.toarray(), T_pauli_ref.toarray()), (
        "pure-Pauli limit failed"
    )

    # alpha_pauli=0, alpha_rmsd=1 → pure RMSD
    T_pure_rmsd, _ = hybrid_coupling_matrix(
        bs, coords, g=0.3, n_qubits=4,
        alpha_pauli=0.0, alpha_rmsd=1.0,
    )
    T_rmsd_ref, _ = rmsd_kernel_coupling(coords)
    assert np.allclose(T_pure_rmsd.toarray(), T_rmsd_ref.toarray()), (
        "pure-RMSD limit failed"
    )

    # 50/50 mix is the average
    T_mix, info = hybrid_coupling_matrix(
        bs, coords, g=0.3, n_qubits=4,
        alpha_pauli=0.5, alpha_rmsd=0.5,
    )
    T_mix_expected = 0.5 * T_pauli_ref + 0.5 * T_rmsd_ref
    assert np.allclose(T_mix.toarray(), T_mix_expected.toarray()), (
        "50/50 hybrid mix failed"
    )
    assert info["mode"] == "hybrid"
    print("[hybrid_coupling] self-test PASSED")


if __name__ == "__main__":
    _self_test()


__all__ = ["hybrid_coupling_matrix"]
