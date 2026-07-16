# Author: Yuqi Zhang
"""Submission helpers — backend factory + dry-run preflight."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from qiskit import QuantumCircuit

from ras_folding.quantum.aer_backend import (
    AerQuantumBackend,
    _circuit_meta,
    _config_to_dict,
    _write_json,
)
from ras_folding.quantum.backend_config import QuantumBackendConfig
from ras_folding.quantum.circuit_builder import QuantumCircuitBuilder
from ras_folding.quantum.ibm_runtime_backend import IBMRuntimeQuantumBackend


def make_quantum_backend(
    config: QuantumBackendConfig,
) -> Union[AerQuantumBackend, IBMRuntimeQuantumBackend]:
    """Factory returning the backend matching `config.backend_type`."""
    if config.is_aer:
        return AerQuantumBackend(config)
    if config.is_ibm_runtime:
        return IBMRuntimeQuantumBackend(config)
    raise ValueError(
        f"unsupported backend_type {config.backend_type!r}"
    )


def prepare_quantum_sampling_run(
    context_or_inputs,
    backend_config: QuantumBackendConfig,
    output_dir: Path,
    n_circuits: int,
    seeds: Optional[Sequence[int]] = None,
    *,
    builder: Optional[QuantumCircuitBuilder] = None,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Pre-flight: build circuits, persist their summary + config.

    This is the canonical dry-run entry point. It NEVER submits anything
    — even when ``backend_config.dry_run`` is False — it only constructs
    the circuits and writes their metadata to disk for inspection.
    To actually submit, call ``make_quantum_backend(config).run_circuits(...)``
    with the circuits returned here.

    Returns
    -------
    dict with keys:
      n_qubits, n_circuits, estimated_total_shots, backend_name,
      execution_mode, circuit_depths, output_dir
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if builder is None:
        builder = QuantumCircuitBuilder()
    circuits: List[QuantumCircuit] = builder.build_circuits(
        context_or_inputs,
        n_circuits=n_circuits,
        seeds=seeds,
        task_id=task_id,
    )

    n_qubits = circuits[0].num_qubits if circuits else 0
    estimated_total_shots = (
        n_circuits * backend_config.shots_per_circuit
    )

    backend_name = (
        "aer_simulator" if backend_config.is_aer
        else backend_config.ibm_backend_name
    )

    summary = {
        "n_qubits": n_qubits,
        "n_circuits": n_circuits,
        "estimated_total_shots": estimated_total_shots,
        "backend_name": backend_name,
        "execution_mode": backend_config.execution_mode,
        "shots_per_circuit": backend_config.shots_per_circuit,
        "max_circuits_per_job": backend_config.max_circuits_per_job,
        "optimization_level": backend_config.optimization_level,
        "circuit_depths": [int(c.depth()) for c in circuits],
        "output_dir": str(output_dir),
    }

    _write_json(output_dir / "quantum_config.json", _config_to_dict(backend_config))
    _write_json(
        output_dir / "quantum_circuits_summary.json",
        {
            "n_circuits": len(circuits),
            "circuits": [_circuit_meta(c) for c in circuits],
        },
    )
    _write_json(output_dir / "preflight_summary.json", summary)

    summary["circuits"] = circuits  # caller may reuse them
    return summary


__all__ = ["make_quantum_backend", "prepare_quantum_sampling_run"]
