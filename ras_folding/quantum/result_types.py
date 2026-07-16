# Author: Yuqi Zhang
"""Quantum-backend result dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_VALID_STATUS = (
    "pending", "submitted", "done", "failed", "dry_run", "skipped",
)


@dataclass
class QuantumCircuitCounts:
    """Per-circuit measurement counts.

    `counts` keys are bitstring strings AS RETURNED by the backend
    (typically Qiskit big-endian: leftmost char = highest qubit). The
    counts_adapter is responsible for any translation needed before
    handing the data to the decoder.
    """
    circuit_name: str
    counts: Dict[str, int]
    shots: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantumBackendResult:
    backend_type: str
    backend_name: str
    execution_mode: str
    status: str
    circuit_counts: List[QuantumCircuitCounts]
    job_ids: List[str]
    output_files: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUS:
            raise ValueError(
                f"status must be one of {_VALID_STATUS}; got {self.status!r}"
            )

    # ------------------------------------------------------------------ #
    def total_shots(self) -> int:
        return int(sum(c.shots for c in self.circuit_counts))


__all__ = ["QuantumCircuitCounts", "QuantumBackendResult"]
