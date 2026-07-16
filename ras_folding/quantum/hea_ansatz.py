# Author: Yuqi Zhang
"""Hardware-Efficient Ansatz (HEA) with bond-block-aware entanglement.

Production quantum state preparation. θ parameters
are NOT optimized via VQE; they come from
``MomentMatchInitializer.compute_theta()`` (closed-form derivation from
V2 corridor prior + 2-point correlations).

Qubit layout
------------
For a fragment with ``n_bonds`` bonds, qubits are arranged as
``[bond_0 (6q)][bond_1 (6q)] ... [bond_{n-1} (6q)]``, so
``n_qubits = 6 * n_bonds``.

Layer structure
---------------
For R rep layers, the circuit is:

    Ry(θ_0) → CX_brick → Ry(θ_1) → CX_brick → ... → Ry(θ_R) → measure(Z)

Total Ry parameters: ``(R + 1) * n_qubits``.

CX brick-wall topology (per layer)
----------------------------------
Two components per layer:

1. **bond-internal**: within each 6-qubit bond block, a brick-wall of CX
   gates connecting adjacent qubits (even pairs then odd pairs).
2. **bond-bridge**: a single CX between the last qubit of bond k and the
   first qubit of bond k+1, for all k = 0..n_bonds-2.

This topology is designed to be ``ibm_cleveland``-friendly: keeps
2-qubit gates between physically-adjacent encoder bits AND across
bond boundaries (matching the autoregressive lattice structure).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from qiskit import QuantumCircuit


N_BITS_PER_BOND = 6


# ---------------------------------------------------------------------- #
# Parameter / topology helpers                                           #
# ---------------------------------------------------------------------- #

def n_parameters(n_qubits: int, reps: int) -> int:
    """Total Ry parameters in the HEA = (reps + 1) * n_qubits."""
    if n_qubits <= 0:
        raise ValueError(f"n_qubits must be > 0; got {n_qubits}")
    if reps < 1:
        raise ValueError(f"reps must be >= 1; got {reps}")
    return (reps + 1) * int(n_qubits)


def cx_edges_bond_aware(
    n_qubits: int, n_bonds: int,
) -> List[Tuple[int, int]]:
    """Return the list of (control, target) CX pairs for ONE brick-wall layer.

    Two parts:
      - bond-internal brick-wall within each 6-qubit block
      - bond-bridge between last qubit of bond k and first of bond k+1

    Edges are returned in execution order (matters for transpile; bonds
    are processed in increasing order). Within each bond, even pairs
    come first, then odd pairs (canonical brick-wall).
    """
    if n_bonds <= 0:
        raise ValueError(f"n_bonds must be > 0; got {n_bonds}")
    if n_qubits != n_bonds * N_BITS_PER_BOND:
        raise ValueError(
            f"n_qubits ({n_qubits}) must equal n_bonds ({n_bonds}) "
            f"* {N_BITS_PER_BOND} = {n_bonds * N_BITS_PER_BOND}"
        )

    edges: List[Tuple[int, int]] = []

    # Bond-internal brick-wall
    for b in range(n_bonds):
        base = b * N_BITS_PER_BOND
        # Even pairs: (0,1) (2,3) (4,5)
        for i in range(0, N_BITS_PER_BOND - 1, 2):
            edges.append((base + i, base + i + 1))
        # Odd pairs: (1,2) (3,4)
        for i in range(1, N_BITS_PER_BOND - 1, 2):
            edges.append((base + i, base + i + 1))

    # Bond-bridge: connect consecutive bonds
    for b in range(n_bonds - 1):
        last_of_b = b * N_BITS_PER_BOND + (N_BITS_PER_BOND - 1)
        first_of_next = (b + 1) * N_BITS_PER_BOND
        edges.append((last_of_b, first_of_next))

    return edges


def count_cx_per_layer(n_qubits: int, n_bonds: int) -> int:
    """Convenience: number of CX gates in one brick-wall layer."""
    return len(cx_edges_bond_aware(n_qubits, n_bonds))


def count_cx_total(n_qubits: int, n_bonds: int, reps: int) -> int:
    """Total CX count across all reps layers."""
    return reps * count_cx_per_layer(n_qubits, n_bonds)


# ---------------------------------------------------------------------- #
# Circuit construction                                                   #
# ---------------------------------------------------------------------- #

def build_hea_circuit(
    n_qubits: int, n_bonds: int, reps: int = 1,
    theta: Optional[np.ndarray] = None,
    name: Optional[str] = None,
) -> QuantumCircuit:
    """Build the HEA circuit with measurement.

    Parameters
    ----------
    n_qubits : circuit width (must = n_bonds * 6)
    n_bonds  : number of bond-blocks
    reps     : number of (CX layer + Ry layer) repeats. The full circuit
               has (reps + 1) Ry layers (initial + reps follow-up) and
               reps CX brick-wall layers.
    theta    : 1-D array of length ``(reps + 1) * n_qubits``. If None,
               an all-zero array is used (no-op state preparation; will
               produce |0...0⟩ when measured).
    name     : optional circuit name

    Returns
    -------
    QuantumCircuit with n_qubits quantum + n_qubits classical registers.
    Measurement basis is Z (standard).
    """
    n_params = n_parameters(n_qubits, reps)
    if theta is None:
        theta = np.zeros(n_params, dtype=np.float64)
    else:
        theta = np.asarray(theta, dtype=np.float64).ravel()
        if theta.size != n_params:
            raise ValueError(
                f"theta length {theta.size} != expected {n_params} "
                f"for n_qubits={n_qubits}, reps={reps}"
            )

    qc_name = name or f"hea_n{n_qubits}_R{reps}"
    qc = QuantumCircuit(n_qubits, n_qubits, name=qc_name)
    cx_edges = cx_edges_bond_aware(n_qubits, n_bonds)

    # First Ry layer
    for q in range(n_qubits):
        qc.ry(float(theta[q]), q)

    # reps × (CX brick-wall + Ry layer)
    for layer_idx in range(reps):
        for (ctrl, tgt) in cx_edges:
            qc.cx(ctrl, tgt)
        offset = (layer_idx + 1) * n_qubits
        for q in range(n_qubits):
            qc.ry(float(theta[offset + q]), q)

    # Measurement (Z basis)
    for q in range(n_qubits):
        qc.measure(q, q)

    # Metadata
    qc.metadata = {
        "ansatz": "hea_with_tf",
        "n_qubits": int(n_qubits),
        "n_bonds": int(n_bonds),
        "reps": int(reps),
        "n_parameters": int(n_params),
        "n_cx_per_layer": int(count_cx_per_layer(n_qubits, n_bonds)),
        "n_cx_total": int(count_cx_total(n_qubits, n_bonds, reps)),
        "theta_norm": float(np.linalg.norm(theta)),
    }
    return qc


# ---------------------------------------------------------------------- #
# Self-test                                                              #
# ---------------------------------------------------------------------- #

def _self_test() -> None:
    """Basic sanity: build, count, run on Aer with small case."""
    # n=3 case: 2 bonds, 12 qubits
    n_qubits = 12
    n_bonds = 2
    reps = 1

    assert n_parameters(n_qubits, reps) == 24
    edges = cx_edges_bond_aware(n_qubits, n_bonds)
    # bond-internal: 5 per bond × 2 bonds = 10
    # bond-bridge: 1
    # total: 11
    assert len(edges) == 11, f"unexpected #edges: {len(edges)}"

    theta = np.random.uniform(-np.pi, np.pi, 24)
    qc = build_hea_circuit(n_qubits, n_bonds, reps=reps, theta=theta)
    assert qc.num_qubits == 12
    assert qc.num_clbits == 12

    # Smoke run on Aer
    try:
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        sim = AerSimulator(method="matrix_product_state")
        tqc = transpile(qc, optimization_level=1)
        result = sim.run(tqc, shots=200, seed_simulator=42).result()
        counts = result.get_counts()
        total = sum(counts.values())
        assert total == 200
        print(f"[hea_ansatz] self-test PASSED "
              f"(n_qubits={n_qubits}, n_bonds={n_bonds}, reps={reps}, "
              f"|theta|={np.linalg.norm(theta):.3f}, "
              f"unique bitstrings sampled = {len(counts)})")
    except ImportError:
        print("[hea_ansatz] self-test SKIPPED (qiskit-aer not available)")


if __name__ == "__main__":
    _self_test()


__all__ = [
    "N_BITS_PER_BOND",
    "n_parameters",
    "cx_edges_bond_aware",
    "count_cx_per_layer",
    "count_cx_total",
    "build_hea_circuit",
]
