# Author: Yuqi Zhang
"""QuantumBackendConfig — configuration for the quantum sampling backend.

Backends supported:
  - "aer_simulator"   — local qiskit-aer
  - "ibm_runtime"     — IBM Runtime SamplerV2 (job mode or batch mode)

Default real-QPU backend name is "ibm_cleveland" — change via
``ibm_backend_name``. IBM credentials are NEVER read from this object.
They must come from the user's saved QiskitRuntimeService account
(via ``QiskitRuntimeService.save_account(...)`` once) or from the
canonical IBM Runtime environment variables. See submission.py /
ibm_runtime_backend.py for resolution order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


_VALID_BACKEND_TYPES = ("aer_simulator", "ibm_runtime")
_VALID_EXECUTION_MODES = ("job", "batch")


@dataclass(frozen=True)
class QuantumBackendConfig:
    """Backend configuration. Frozen — pass a new instance to override."""

    backend_type: str = "aer_simulator"
    ibm_backend_name: str = "ibm_cleveland"
    ibm_channel: Optional[str] = None
    ibm_instance: Optional[str] = None

    execution_mode: str = "batch"  # "job" or "batch"
    shots_per_circuit: int = 2048
    max_circuits_per_job: int = 100
    optimization_level: int = 3
    seed_simulator: Optional[int] = None
    seed_transpiler: Optional[int] = None

    resilience_level: Optional[int] = None
    default_shots: Optional[int] = None

    dry_run: bool = False
    overwrite: bool = False

    # When True (default), every call into QuantumBackendBaseSampler.sample
    # uses the SAME seed_simulator / circuit-builder seeds — i.e. each
    # tau sees identical raw counts. This is the right setting for
    # filter-vs-full-energy diagnostics (every tau is filtered against
    # the same distribution).
    #
    # When False, each call derives a fresh seed_simulator AND fresh
    # circuit-builder seeds from the per-call ``seed`` argument. This is
    # the right setting for production runs where each tau should sample
    # from an independent draw.
    paired_tau_sampling: bool = True

    # --- IBM Runtime job-level load balancing (V2) ---------------------
    # All four bounds are independently optional. The chunk planner in
    # ibm_runtime_backend.plan_runtime_chunks() applies whichever ones
    # are set; an unset bound is simply not enforced. V1 callers that
    # never touch these keep the old "by_circuit_count" behaviour.
    max_shots_per_job: Optional[int] = None
    max_estimated_runtime_sec_per_job: Optional[float] = None
    estimated_sec_per_shot: Optional[float] = None
    estimated_job_overhead_sec: Optional[float] = 30.0
    max_chunks_per_task: Optional[int] = None
    chunk_strategy: str = "by_circuit_count"  # default keeps V1 behaviour
    allow_oversized_job: bool = False

    def __post_init__(self) -> None:
        if self.backend_type not in _VALID_BACKEND_TYPES:
            raise ValueError(
                f"backend_type must be one of {_VALID_BACKEND_TYPES}; "
                f"got {self.backend_type!r}"
            )
        if self.execution_mode not in _VALID_EXECUTION_MODES:
            raise ValueError(
                f"execution_mode must be one of {_VALID_EXECUTION_MODES}; "
                f"got {self.execution_mode!r}"
            )
        if self.shots_per_circuit <= 0:
            raise ValueError(
                f"shots_per_circuit must be > 0; got {self.shots_per_circuit}"
            )
        if self.max_circuits_per_job <= 0:
            raise ValueError(
                f"max_circuits_per_job must be > 0; got {self.max_circuits_per_job}"
            )
        if self.optimization_level not in (0, 1, 2, 3):
            raise ValueError(
                f"optimization_level must be 0/1/2/3; got {self.optimization_level}"
            )
        valid_chunk_strategies = (
            "by_circuit_count", "by_shots", "balanced",
        )
        if self.chunk_strategy not in valid_chunk_strategies:
            raise ValueError(
                f"chunk_strategy must be one of "
                f"{valid_chunk_strategies}; got {self.chunk_strategy!r}"
            )
        if (self.max_shots_per_job is not None
                and self.max_shots_per_job <= 0):
            raise ValueError(
                f"max_shots_per_job must be > 0 when set; "
                f"got {self.max_shots_per_job}"
            )
        if (self.max_estimated_runtime_sec_per_job is not None
                and self.max_estimated_runtime_sec_per_job <= 0):
            raise ValueError(
                "max_estimated_runtime_sec_per_job must be > 0 when set"
            )

    # ------------------------------------------------------------------ #
    @property
    def is_aer(self) -> bool:
        return self.backend_type == "aer_simulator"

    @property
    def is_ibm_runtime(self) -> bool:
        return self.backend_type == "ibm_runtime"


__all__ = ["QuantumBackendConfig"]
