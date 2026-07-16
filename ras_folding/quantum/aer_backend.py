# Author: Yuqi Zhang
"""AerQuantumBackend — local qiskit-aer simulator backend.

Reads circuits from an iterable of ``QuantumCircuit`` (already built by
QuantumCircuitBuilder), transpiles, runs, and persists:

  output_dir/
    quantum_config.json
    quantum_circuits_summary.json
    transpile_summary.json
    raw_counts.json
    backend_result.json

dry_run=True writes only quantum_config.json + quantum_circuits_summary.json,
returns a result with status="dry_run".

If qiskit-aer isn't installed, run_circuits raises a clear ImportError.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from qiskit import QuantumCircuit, transpile

from ras_folding.quantum.backend_config import QuantumBackendConfig
from ras_folding.quantum.result_types import (
    QuantumBackendResult,
    QuantumCircuitCounts,
)


_AER_BACKEND_NAME = "aer_simulator"


class AerQuantumBackend:
    def __init__(self, config: QuantumBackendConfig) -> None:
        if not config.is_aer:
            raise ValueError(
                f"AerQuantumBackend requires backend_type='aer_simulator'; "
                f"got {config.backend_type!r}"
            )
        self.config = config

    # ------------------------------------------------------------------ #
    def run_circuits(
        self,
        circuits: Sequence[QuantumCircuit],
        output_dir: Path,
    ) -> QuantumBackendResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        files: Dict[str, str] = {}

        # Always persist config + circuit summary (even on dry_run / failure)
        files["quantum_config.json"] = _write_json(
            output_dir / "quantum_config.json",
            _config_to_dict(self.config),
        )
        files["quantum_circuits_summary.json"] = _write_json(
            output_dir / "quantum_circuits_summary.json",
            {
                "n_circuits": len(circuits),
                "circuits": [_circuit_meta(c) for c in circuits],
            },
        )

        if self.config.dry_run:
            res = QuantumBackendResult(
                backend_type="aer_simulator",
                backend_name=_AER_BACKEND_NAME,
                execution_mode="job",
                status="dry_run",
                circuit_counts=[],
                job_ids=[],
                output_files=files,
                metadata={
                    "n_circuits": len(circuits),
                    "shots_per_circuit": self.config.shots_per_circuit,
                },
            )
            files["backend_result.json"] = _write_json(
                output_dir / "backend_result.json",
                _result_to_dict(res),
            )
            res.output_files = files
            return res

        # --- import qiskit-aer lazily ----------------------------------
        try:
            from qiskit_aer import AerSimulator
        except ImportError as e:  # pragma: no cover
            err = (
                "qiskit-aer is not installed. "
                "Install with `pip install qiskit-aer` to run AER backend."
            )
            (output_dir / "ERROR.txt").write_text(err + "\n")
            raise ImportError(err) from e

        try:
            # Use matrix_product_state for arbitrary qubit counts. Our
            # circuits have ONLY single-qubit gates + measurements (no
            # entangling layer), so the MPS bond dimension stays at 1
            # and each shot costs O(N) memory rather than O(2^N).
            # Statevector would OOM for n_qubits > ~30.
            sim = AerSimulator(method="matrix_product_state")

            # NOTE: do NOT pass `backend=sim` here. qiskit-aer >= 0.17
            # exposes a default Target with a 31-qubit coupling map even
            # for the abstract AerSimulator(); transpile-against-backend
            # would then reject any circuit with > 31 qubits. AerSimulator
            # itself (run path) has no such limit. We keep optimization
            # passes (optimization_level / seed_transpiler) but drop the
            # synthetic coupling-map constraint.
            t_circuits = transpile(
                list(circuits),
                optimization_level=self.config.optimization_level,
                seed_transpiler=self.config.seed_transpiler,
            )
            files["transpile_summary.json"] = _write_json(
                output_dir / "transpile_summary.json",
                {
                    "optimization_level": self.config.optimization_level,
                    "seed_transpiler": self.config.seed_transpiler,
                    "circuits": [
                        {
                            "name": c.name,
                            "depth": c.depth(),
                            "num_qubits": c.num_qubits,
                            "num_clbits": c.num_clbits,
                        }
                        for c in t_circuits
                    ],
                },
            )

            run_kwargs: Dict[str, Any] = {
                "shots": self.config.shots_per_circuit,
            }
            if self.config.seed_simulator is not None:
                run_kwargs["seed_simulator"] = self.config.seed_simulator

            job = sim.run(t_circuits, **run_kwargs)
            result = job.result()

            circuit_counts: List[QuantumCircuitCounts] = []
            # Use index-based result lookup. The transpile pass may rename
            # circuits internally; matching by circuit object or .name is
            # not robust across qiskit-aer versions. result.get_counts()
            # returns a list for multi-circuit jobs and a single dict for
            # single-circuit jobs.
            all_counts = result.get_counts()
            if isinstance(all_counts, dict):
                all_counts = [all_counts]
            if len(all_counts) != len(circuits):
                raise RuntimeError(
                    f"AER result returned {len(all_counts)} count blocks "
                    f"for {len(circuits)} circuits"
                )
            for src_qc, counts in zip(circuits, all_counts):
                circuit_counts.append(QuantumCircuitCounts(
                    circuit_name=src_qc.name,
                    counts=dict(counts),
                    shots=self.config.shots_per_circuit,
                    metadata=dict(src_qc.metadata or {}),
                ))

            res = QuantumBackendResult(
                backend_type="aer_simulator",
                backend_name=_AER_BACKEND_NAME,
                execution_mode="job",
                status="done",
                circuit_counts=circuit_counts,
                job_ids=[],
                output_files=files,
                metadata={
                    "n_circuits": len(circuits),
                    "shots_per_circuit": self.config.shots_per_circuit,
                    "total_shots": (
                        len(circuits) * self.config.shots_per_circuit
                    ),
                    "seed_simulator": self.config.seed_simulator,
                    "seed_transpiler": self.config.seed_transpiler,
                },
            )
        except Exception as e:
            err = f"AER run failed: {e!r}"
            (output_dir / "ERROR.txt").write_text(err + "\n")
            res = QuantumBackendResult(
                backend_type="aer_simulator",
                backend_name=_AER_BACKEND_NAME,
                execution_mode="job",
                status="failed",
                circuit_counts=[],
                job_ids=[],
                output_files=files,
                error=err,
                metadata={"n_circuits": len(circuits)},
            )

        # persist counts + result
        files["raw_counts.json"] = _write_json(
            output_dir / "raw_counts.json",
            _raw_counts_dict(res),
        )
        files["backend_result.json"] = _write_json(
            output_dir / "backend_result.json",
            _result_to_dict(res),
        )
        res.output_files = files
        return res


# ---------------------------------------------------------------------- #
# helpers shared by aer / ibm backends                                   #
# ---------------------------------------------------------------------- #

def _write_json(path: Path, payload: Any) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    return str(path)


def _json_default(o: Any) -> Any:
    """Best-effort fallback serializer for things json doesn't know."""
    if hasattr(o, "isoformat"):
        return o.isoformat()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return repr(o)


def _config_to_dict(cfg: QuantumBackendConfig) -> Dict[str, Any]:
    return asdict(cfg)


def _circuit_meta(c: QuantumCircuit) -> Dict[str, Any]:
    return {
        "name": c.name,
        "num_qubits": c.num_qubits,
        "num_clbits": c.num_clbits,
        "depth_pre_transpile": c.depth(),
        "metadata": dict(c.metadata or {}),
    }


def _raw_counts_dict(res: QuantumBackendResult) -> Dict[str, Any]:
    return {
        "backend_type": res.backend_type,
        "backend_name": res.backend_name,
        "execution_mode": res.execution_mode,
        "job_ids": list(res.job_ids),
        "circuits": [
            {
                "circuit_name": c.circuit_name,
                "shots": c.shots,
                "counts": dict(c.counts),
                "metadata": dict(c.metadata),
            }
            for c in res.circuit_counts
        ],
    }


def _result_to_dict(res: QuantumBackendResult) -> Dict[str, Any]:
    return {
        "backend_type": res.backend_type,
        "backend_name": res.backend_name,
        "execution_mode": res.execution_mode,
        "status": res.status,
        "job_ids": list(res.job_ids),
        "n_circuits": len(res.circuit_counts),
        "total_shots": res.total_shots(),
        "error": res.error,
        "metadata": dict(res.metadata),
    }


__all__ = ["AerQuantumBackend"]
