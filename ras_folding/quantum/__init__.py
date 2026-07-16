# Author: Yuqi Zhang
"""ras_folding.quantum — quantum sampling backends + circuit / counts plumbing.

Public API:
  QuantumBackendConfig                   -- backend configuration dataclass
  QuantumCircuitCounts, QuantumBackendResult
                                         -- result dataclasses
  QuantumCircuitBuilder                  -- build measurement circuits
  counts_to_candidate_samples            -- counts → CandidateSample
  AerQuantumBackend                      -- AER local simulator backend
  IBMRuntimeQuantumBackend               -- IBM Runtime SamplerV2 backend
  make_quantum_backend                   -- factory
  prepare_quantum_sampling_run           -- dry-run / pre-flight helper
"""
from ras_folding.quantum.backend_config import QuantumBackendConfig
from ras_folding.quantum.result_types import (
    QuantumBackendResult,
    QuantumCircuitCounts,
)
from ras_folding.quantum.circuit_builder import QuantumCircuitBuilder
from ras_folding.quantum.counts_adapter import counts_to_candidate_samples
from ras_folding.quantum.aer_backend import AerQuantumBackend
from ras_folding.quantum.ibm_runtime_backend import IBMRuntimeQuantumBackend
from ras_folding.quantum.submission import (
    make_quantum_backend,
    prepare_quantum_sampling_run,
)

__all__ = [
    "QuantumBackendConfig",
    "QuantumCircuitCounts",
    "QuantumBackendResult",
    "QuantumCircuitBuilder",
    "counts_to_candidate_samples",
    "AerQuantumBackend",
    "IBMRuntimeQuantumBackend",
    "make_quantum_backend",
    "prepare_quantum_sampling_run",
]
