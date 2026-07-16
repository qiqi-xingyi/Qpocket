# Author: Yuqi Zhang
"""QuantumCircuitBuilder — generate measurement circuits for base sampling.

This module builds the circuits used to draw raw bitstring samples from
the quantum backend. In this ``full_pipline`` (single-flow) build it
implements a SINGLE ansatz — the Hardware-Efficient Ansatz with
transverse-field-style brick-wall entanglement (``hea_with_tf``). There
is no VQE / QAOA / parametric training: the HEA parameters θ are derived
in closed form by ``MomentMatchInitializer`` and passed in.

Qubit-to-bitstring convention
-----------------------------
For a fragment with ``n_bonds`` bonds and ``bits_per_step`` bits per
bond, we create ``n_qubits = n_bonds * bits_per_step`` qubits with the
following layout:

    qubit_index = bond_index * bits_per_step + bit_offset_in_bond

where ``bit_offset_in_bond = 0`` is the MSB of that bond's 6-bit code.

After measurement, Qiskit returns counts whose keys are bitstrings with
the LEFTMOST character corresponding to the HIGHEST qubit index. To
recover the decoder's MSB-first / bond-0-first convention,
``counts_adapter`` reverses this string. This convention is documented
in counts_adapter.counts_to_candidate_samples.

Ansatz
------
- ``hea_with_tf`` (the only ansatz): genuine-entanglement state
  preparation. The full θ vector (length ``(reps_hea + 1) * n_qubits``)
  is produced by ``MomentMatchInitializer.compute_theta()`` and supplied
  via ``ansatz_params``. Circuit assembly is delegated to
  ``ras_folding.quantum.hea_ansatz.build_hea_circuit``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from qiskit import QuantumCircuit

from ras_folding.encoder.decoder import BITS_PER_BOND
from ras_folding.sampler.context import get_encoder_inputs


_VALID_ANSATZE = ("hea_with_tf",)


class QuantumCircuitBuilder:
    """Build measurement circuits for a fragment (``hea_with_tf``).

    Parameters
    ----------
    ansatz : "hea_with_tf" (the only supported value).
    reps : int — number of repetitions of the base layer (>=1). Retained
        for circuit metadata; the HEA depth is governed by ``reps_hea``.
    seed : int | None — RNG seed for builder-side per-circuit seed draws.
    ansatz_params : np.ndarray | None — the full HEA θ vector
        (length ``(reps_hea + 1) * n_qubits``) produced by
        ``MomentMatchInitializer``. May be ``None`` at construction time
        (e.g. for a placeholder builder); it is then required at
        ``build_circuits`` time, which raises a clear error if missing.
    reps_hea : int — number of CX brick-wall + Ry layers in the HEA.
    """

    def __init__(
        self,
        ansatz: str = "hea_with_tf",
        reps: int = 1,
        seed: Optional[int] = None,
        *,
        ansatz_params: Optional[np.ndarray] = None,
        reps_hea: int = 1,
    ) -> None:
        if ansatz not in _VALID_ANSATZE:
            raise ValueError(
                f"ansatz must be one of {_VALID_ANSATZE}; got {ansatz!r}"
            )
        if reps < 1:
            raise ValueError(f"reps must be >= 1; got {reps}")
        self.ansatz = ansatz
        self.reps = int(reps)
        self.seed = seed
        # hea_with_tf params. ansatz_params may legitimately be None here
        # (a placeholder builder); the requirement is enforced at build
        # time in build_circuits(), so a param-less builder can still be
        # constructed by callers that supply θ later.
        if ansatz_params is not None:
            arr = np.asarray(ansatz_params, dtype=np.float64).ravel()
            if not np.all(np.isfinite(arr)):
                raise ValueError("ansatz_params contains non-finite values")
            self.ansatz_params = arr
        else:
            self.ansatz_params = None
        if reps_hea < 1:
            raise ValueError(f"reps_hea must be >= 1; got {reps_hea}")
        self.reps_hea = int(reps_hea)

    # ------------------------------------------------------------------ #
    def build_circuits(
        self,
        context_or_inputs,
        n_circuits: int,
        seeds: Optional[Sequence[int]] = None,
        *,
        task_id: Optional[str] = None,
        bits_per_step: int = BITS_PER_BOND,
    ) -> List[QuantumCircuit]:
        if n_circuits <= 0:
            raise ValueError(f"n_circuits must be > 0; got {n_circuits}")
        encoder_inputs = get_encoder_inputs(context_or_inputs)
        n_bonds = int(encoder_inputs.n_bonds)
        n_qubits = n_bonds * int(bits_per_step)
        if n_qubits == 0:
            raise ValueError("circuit has 0 qubits (n_bonds == 0)")

        if seeds is None:
            rng = np.random.default_rng(self.seed)
            seeds = rng.integers(
                low=0, high=np.iinfo(np.int64).max,
                size=n_circuits, dtype=np.int64,
            ).tolist()
        else:
            seeds = list(seeds)
            if len(seeds) != n_circuits:
                raise ValueError(
                    f"seeds length {len(seeds)} != n_circuits {n_circuits}"
                )

        # θ is pre-computed externally (MomentMatchInitializer)
        # and shared across all circuits for this task.
        expected_len = (self.reps_hea + 1) * n_qubits
        if self.ansatz_params is None:
            raise ValueError(
                "ansatz='hea_with_tf' requires ansatz_params (from "
                "MomentMatchInitializer.compute_theta()) before building "
                "circuits"
            )
        if self.ansatz_params.size != expected_len:
            raise ValueError(
                f"ansatz_params length {self.ansatz_params.size} != "
                f"expected (reps_hea+1)*n_qubits = "
                f"{self.reps_hea + 1} * {n_qubits} = {expected_len}"
            )
        thetas = self.ansatz_params
        prior_meta: Dict[str, Any] = {
            "prior_source": "moment_match_v2",
            "reps_hea": int(self.reps_hea),
            "theta_norm": float(np.linalg.norm(thetas)),
            "n_parameters": int(thetas.size),
        }

        out: List[QuantumCircuit] = []
        for idx in range(n_circuits):
            qc = self._build_one(
                n_qubits=n_qubits,
                seed=int(seeds[idx]),
                circuit_index=idx,
                task_id=task_id,
                n_bonds=n_bonds,
                bits_per_step=int(bits_per_step),
                thetas=thetas,
                prior_meta=prior_meta,
            )
            out.append(qc)
        return out

    # ------------------------------------------------------------------ #
    def _build_one(
        self,
        *,
        n_qubits: int,
        seed: int,
        circuit_index: int,
        task_id: Optional[str],
        n_bonds: int,
        bits_per_step: int,
        thetas: np.ndarray,
        prior_meta: Dict[str, Any],
    ) -> QuantumCircuit:
        tid = task_id if task_id is not None else "task"
        name = f"{tid}__{self.ansatz}__nq{n_qubits}__c{circuit_index}__s{seed}"

        # delegate to hea_ansatz.build_hea_circuit which also adds
        # measurement gates and sets HEA-specific metadata.
        from ras_folding.quantum.hea_ansatz import build_hea_circuit
        qc = build_hea_circuit(
            n_qubits=n_qubits, n_bonds=n_bonds,
            reps=self.reps_hea, theta=thetas,
            name=name,
        )
        hea_meta = dict(qc.metadata or {})

        meta: Dict[str, Any] = {
            "task_id": tid,
            "circuit_index": circuit_index,
            "seed": int(seed),
            "ansatz": self.ansatz,
            "reps": self.reps,
            "n_qubits": int(n_qubits),
            "n_bonds": int(n_bonds),
            "bits_per_step": int(bits_per_step),
        }
        if prior_meta:
            meta.update(prior_meta)
        if hea_meta:
            meta.update(hea_meta)
        qc.metadata = meta
        return qc


__all__ = ["QuantumCircuitBuilder"]
