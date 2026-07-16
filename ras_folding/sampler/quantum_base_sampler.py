# Author: Yuqi Zhang
"""QuantumBackendBaseSampler — drop-in replacement for EncoderBaseSampler
that draws raw bitstrings from a configured quantum backend.

Pipeline:
    build circuits → run on backend → counts_to_candidate_samples
                                        → list[CandidateSample]

The samples returned have ``coords=None``, ``valid=False``,
``accepted=False`` — exactly matching EncoderBaseSampler. Downstream
QuantumImaginaryTimeSampler / decode_and_validate / FilterHamiltonian /
FullEnergyScorer work unchanged.

n_samples semantics
-------------------
For a quantum backend, the actual sample count is the SUM of measurement
counts across all circuits, which equals
    n_circuits * shots_per_circuit
NOT necessarily ``n_samples``. We surface this honestly:
  - ``n_samples`` is recorded in metadata as ``requested_n_samples``;
  - ``total_shots`` is the actual count budget;
  - if ``n_samples`` is given and disagrees, metadata records
    ``n_samples_mismatch=True`` but no exception is raised (we do not
    fabricate "exactly n_samples" candidates from quantum counts).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from ras_folding.quantum.backend_config import QuantumBackendConfig
from ras_folding.quantum.circuit_builder import QuantumCircuitBuilder
from ras_folding.quantum.counts_adapter import counts_to_candidate_samples
from ras_folding.quantum.submission import make_quantum_backend
from ras_folding.sampler.context import get_encoder_inputs
from ras_folding.sampler.sample_types import CandidateSample


class QuantumBackendBaseSampler:
    """Quantum-backed base sampler. Same contract as EncoderBaseSampler.sample."""

    def __init__(
        self,
        backend_config: QuantumBackendConfig,
        circuit_builder: Optional[QuantumCircuitBuilder] = None,
        n_circuits: int = 8,
        seeds: Optional[Sequence[int]] = None,
        output_dir: Optional[Path] = None,
        task_id: Optional[str] = None,
    ) -> None:
        if n_circuits <= 0:
            raise ValueError(f"n_circuits must be > 0; got {n_circuits}")
        self.backend_config = backend_config
        self.circuit_builder = circuit_builder or QuantumCircuitBuilder()
        self.n_circuits = int(n_circuits)
        self.seeds = list(seeds) if seeds is not None else None
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.task_id = task_id

    # ------------------------------------------------------------------ #
    def sample(
        self,
        context_or_inputs,
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[CandidateSample]:
        encoder_inputs = get_encoder_inputs(context_or_inputs)
        # determine output_dir
        if self.output_dir is None:
            raise ValueError(
                "QuantumBackendBaseSampler requires an output_dir "
                "(set via __init__ or wire it through your runner)."
            )

        # --- paired_tau_sampling resolution ---------------------------
        paired = self.backend_config.paired_tau_sampling
        if paired:
            # All calls reuse config seeds → identical counts across calls.
            backend_config_for_run = self.backend_config
            seeds_to_use = self.seeds  # may be None — builder draws from its own RNG
            effective_seed_simulator = self.backend_config.seed_simulator
            # Same output_dir for all calls (the data is identical).
            out_dir = self.output_dir
        else:
            # Each call must derive a fresh seed_simulator AND fresh
            # circuit-builder seeds from the per-call ``seed`` argument.
            if seed is None:
                raise ValueError(
                    "paired_tau_sampling=False requires a per-call seed; "
                    "QuantumImaginaryTimeSampler passes one automatically."
                )
            rng = np.random.default_rng(int(seed))
            effective_seed_simulator = int(
                rng.integers(0, np.iinfo(np.int64).max)
            )
            circuit_root = int(rng.integers(0, np.iinfo(np.int64).max))
            rng2 = np.random.default_rng(circuit_root)
            seeds_to_use = rng2.integers(
                low=0, high=np.iinfo(np.int64).max,
                size=self.n_circuits, dtype=np.int64,
            ).tolist()
            backend_config_for_run = dataclasses.replace(
                self.backend_config,
                seed_simulator=effective_seed_simulator,
            )
            # Per-call sub-directory so artefacts are not overwritten
            # across taus.
            out_dir = self.output_dir / f"seed_{int(seed)}"

        out_dir.mkdir(parents=True, exist_ok=True)

        circuits = self.circuit_builder.build_circuits(
            context_or_inputs,
            n_circuits=self.n_circuits,
            seeds=seeds_to_use,
            task_id=self.task_id,
        )

        backend = make_quantum_backend(backend_config_for_run)
        result = backend.run_circuits(circuits, output_dir=out_dir)

        samples = counts_to_candidate_samples(
            result, context_or_inputs,
        )

        # surface backend metadata + budget on every sample
        total_shots = self.n_circuits * self.backend_config.shots_per_circuit
        n_samples_mismatch = (
            n_samples is not None and int(n_samples) != total_shots
        )
        for s in samples:
            s.metadata.setdefault("backend_type", result.backend_type)
            s.metadata.setdefault("backend_name", result.backend_name)
            s.metadata.setdefault("execution_mode", result.execution_mode)
            s.metadata.setdefault("job_ids", list(result.job_ids))
            s.metadata["total_shots"] = total_shots
            s.metadata["n_circuits"] = self.n_circuits
            s.metadata["shots_per_circuit"] = (
                self.backend_config.shots_per_circuit
            )
            s.metadata["paired_tau_sampling"] = bool(paired)
            s.metadata["effective_seed_simulator"] = effective_seed_simulator
            s.metadata["effective_circuit_seeds"] = (
                None if seeds_to_use is None else [int(x) for x in seeds_to_use]
            )
            s.metadata["effective_output_dir"] = str(out_dir)
            if n_samples is not None:
                s.metadata["requested_n_samples"] = int(n_samples)
                s.metadata["n_samples_mismatch"] = bool(n_samples_mismatch)

        return samples


__all__ = ["QuantumBackendBaseSampler"]
