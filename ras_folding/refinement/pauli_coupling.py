# Author: Yuqi Zhang
"""Pauli matrix-element coupling for the SQD refinement step.

Used by ``SubspaceDiagonalizationRefiner`` when
``coupling_mode in ("pauli_hamming1", "hybrid")``.

Mathematical content
--------------------
The non-diagonal part of the effective Hamiltonian Ĥ = H_filter - g·∑_q X_q
has matrix elements in the sampled-bitstring basis

    ⟨z_i| (-g·∑_q X_q) |z_j⟩  =  -g · 1[popcount(z_i ⊕ z_j) == 1]

i.e. non-zero iff z_i and z_j differ in exactly one bit, and the value
is exactly -g. This is the *exact* Pauli matrix element on the
computational-basis subspace spanned by the sampled bitstrings — there
is no approximation here.

Why this matters
----------------
In the previous "SQD-inspired" implementation the off-diagonal coupling
was an ad-hoc RMSD-Gaussian kernel with no quantum-mechanical interpretation.
Here the effective Hamiltonian Ĥ has a real Pauli decomposition,
and this module gives the exact projection of Ĥ onto the sampled subspace.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------- #
# popcount                                                               #
# ---------------------------------------------------------------------- #

def popcount(x: int) -> int:
    """Number of set bits in integer x. Uses int.bit_count() if available
    (Python 3.10+), else falls back to a portable shift-loop.
    """
    bc = getattr(int, "bit_count", None)
    if bc is not None:
        return x.bit_count()
    # Portable fallback
    c = 0
    while x:
        c += x & 1
        x >>= 1
    return c


def popcount_array(xs: np.ndarray) -> np.ndarray:
    """Vectorised popcount on a numpy int64 array."""
    out = np.zeros_like(xs, dtype=np.int32)
    if hasattr(np, "bitwise_count"):  # numpy >= 2.0
        out = np.bitwise_count(xs).astype(np.int32)
        return out
    # Fallback: shift loop
    xs = xs.astype(np.int64).copy()
    table = np.array([bin(i).count("1") for i in range(256)],
                     dtype=np.int32)
    while np.any(xs):
        out += table[(xs & 0xFF).astype(np.int64)]
        xs >>= 8
    return out


# ---------------------------------------------------------------------- #
# Hamming-1 Pauli coupling                                               #
# ---------------------------------------------------------------------- #

def pauli_hamming1_matrix(
    bitstrings: List[int], g: float, n_qubits: int,
) -> sp.csr_matrix:
    """Build T[i, j] = -g · 1[Hamming(z_i, z_j) = 1] as an N×N sparse matrix.

    This is the exact matrix element ⟨z_i| (-g·∑_q X_q) |z_j⟩ restricted
    to the sampled basis {|z_i⟩}.

    Parameters
    ----------
    bitstrings : list of N integer-encoded bitstrings
    g          : transverse-field coupling (must be > 0)
    n_qubits   : circuit width (for validation only)

    Returns
    -------
    T : (N, N) scipy.sparse.csr_matrix, symmetric, zero diagonal.

    Complexity
    ----------
    O(N²) pairwise XOR + popcount. For N ≤ 500 (typical refinement
    subspace size) this is < 1 ms.
    """
    if g <= 0:
        raise ValueError(f"g must be > 0; got {g}")
    N = len(bitstrings)
    if N == 0:
        return sp.csr_matrix((0, 0), dtype=np.float64)

    bs_arr = np.asarray(bitstrings, dtype=np.int64)
    # mask check: ensure no bitstring exceeds n_qubits bits
    if n_qubits < 63:
        max_val = (1 << n_qubits) - 1
        if bs_arr.max() > max_val:
            raise ValueError(
                f"bitstring exceeds n_qubits={n_qubits} bit width; "
                f"max value = {bs_arr.max()}, allowed = {max_val}"
            )

    # Pairwise XOR via broadcasting (N×N)
    # For N=200 → 40K pairs; for N=500 → 250K pairs. Tractable.
    xor_mat = np.bitwise_xor(bs_arr[:, None], bs_arr[None, :])
    pop_mat = popcount_array(xor_mat.ravel()).reshape(N, N)

    # T[i, j] = -g iff popcount(xor) == 1 (and i != j)
    mask = (pop_mat == 1)
    np.fill_diagonal(mask, False)  # safety, popcount(0) == 0 anyway

    rows, cols = np.where(mask)
    data = np.full(rows.size, -float(g), dtype=np.float64)

    T = sp.csr_matrix(
        (data, (rows, cols)), shape=(N, N), dtype=np.float64,
    )
    # Force exact symmetry (should already be by mask symmetry)
    T = (T + T.T).multiply(0.5).tocsr()
    return T


# ---------------------------------------------------------------------- #
# Diagnostics                                                            #
# ---------------------------------------------------------------------- #

def hamming_graph_stats(
    bitstrings: List[int], n_qubits: int,
) -> dict:
    """Diagnostics: how connected is the sampled subspace in Hamming-1 graph?

    Returns dict with:
      n_pairs_hamming_1 : edge count
      avg_degree        : mean Hamming-1 degree
      max_degree        : max Hamming-1 degree
      n_isolated        : nodes with zero Hamming-1 neighbours
    """
    N = len(bitstrings)
    if N <= 1:
        return {
            "n_pairs_hamming_1": 0,
            "avg_degree": 0.0,
            "max_degree": 0,
            "n_isolated": N,
        }
    bs_arr = np.asarray(bitstrings, dtype=np.int64)
    xor_mat = np.bitwise_xor(bs_arr[:, None], bs_arr[None, :])
    pop_mat = popcount_array(xor_mat.ravel()).reshape(N, N)
    mask = (pop_mat == 1)
    np.fill_diagonal(mask, False)
    degrees = mask.sum(axis=1)
    return {
        "n_pairs_hamming_1": int(mask.sum() // 2),
        "avg_degree": float(degrees.mean()),
        "max_degree": int(degrees.max()),
        "n_isolated": int((degrees == 0).sum()),
    }


# ---------------------------------------------------------------------- #
# Self-test                                                              #
# ---------------------------------------------------------------------- #

def _self_test() -> None:
    """Verify Hamming-1 detection on a hand-crafted example."""
    bs = [0b000000, 0b000001, 0b000010, 0b000011, 0b000100]
    T = pauli_hamming1_matrix(bs, g=0.5, n_qubits=6)
    Td = T.toarray()

    # 0b000000 vs 0b000001: H=1 → -0.5
    assert Td[0, 1] == -0.5
    # 0b000000 vs 0b000010: H=1 → -0.5
    assert Td[0, 2] == -0.5
    # 0b000000 vs 0b000011: H=2 → 0
    assert Td[0, 3] == 0.0
    # 0b000000 vs 0b000100: H=1 → -0.5
    assert Td[0, 4] == -0.5
    # 0b000001 vs 0b000011: H=1 → -0.5
    assert Td[1, 3] == -0.5
    # 0b000010 vs 0b000011: H=1 → -0.5
    assert Td[2, 3] == -0.5
    # Symmetry
    assert np.allclose(Td, Td.T)
    # Zero diagonal
    assert np.all(np.diag(Td) == 0)

    stats = hamming_graph_stats(bs, n_qubits=6)
    assert stats["n_pairs_hamming_1"] >= 4
    print("[pauli_coupling] self-test PASSED")


if __name__ == "__main__":
    _self_test()


__all__ = [
    "popcount", "popcount_array",
    "pauli_hamming1_matrix",
    "hamming_graph_stats",
]
