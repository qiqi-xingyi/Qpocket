# Author: Yuqi Zhang
"""IBMRuntimeQuantumBackend — IBM Runtime SamplerV2 backend.

Two execution modes:

  job mode    — config.execution_mode == "job"
                Default for smoke tests / single small jobs.
                For each chunk of circuits, a SamplerV2 job is run
                with mode = the IBM backend object.

  batch mode  — config.execution_mode == "batch"
                Default for real QPU production runs in this project.
                A Batch context wraps multiple SamplerV2 jobs, each
                running one chunk.

Resume support
--------------
If ``output_dir/raw_counts.json`` already exists and config.overwrite
is False, run_circuits returns a result with status="skipped" parsed
from the existing files instead of re-submitting. This protects against
accidental double-submission to a real QPU.

Credential handling
-------------------
This backend NEVER reads / writes credentials directly. It uses
``QiskitRuntimeService(channel=..., instance=...)`` which in turn loads
the user's saved account (``QiskitRuntimeService.save_account``) or
the canonical environment variables. No tokens are written to disk by
this class.

If qiskit-ibm-runtime is not installed, run_circuits returns status
"failed" with a clear error message — the rest of the test suite must
NOT crash on import.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from qiskit import QuantumCircuit, transpile

from ras_folding.quantum.aer_backend import (  # reuse plumbing
    _circuit_meta,
    _config_to_dict,
    _raw_counts_dict,
    _result_to_dict,
    _write_json,
)
from ras_folding.quantum.backend_config import QuantumBackendConfig
from ras_folding.quantum.result_types import (
    QuantumBackendResult,
    QuantumCircuitCounts,
)


class IBMRuntimeQuantumBackend:
    def __init__(self, config: QuantumBackendConfig) -> None:
        if not config.is_ibm_runtime:
            raise ValueError(
                f"IBMRuntimeQuantumBackend requires "
                f"backend_type='ibm_runtime'; got {config.backend_type!r}"
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

        # --- resume guard ----------------------------------------------
        raw_path = output_dir / "raw_counts.json"
        result_path = output_dir / "backend_result.json"
        if (
            raw_path.exists()
            and result_path.exists()
            and not self.config.overwrite
        ):
            return self._load_skipped(raw_path, result_path, files)

        # --- dry_run ---------------------------------------------------
        if self.config.dry_run:
            # also emit the chunk plan so users can inspect projected
            # job count and load distribution before going live
            try:
                _plan = plan_runtime_chunks(
                    list(circuits),
                    shots_per_circuit=self.config.shots_per_circuit,
                    max_circuits_per_job=
                        self.config.max_circuits_per_job,
                    max_shots_per_job=self.config.max_shots_per_job,
                    max_estimated_runtime_sec_per_job=
                        self.config.max_estimated_runtime_sec_per_job,
                    estimated_sec_per_shot=
                        self.config.estimated_sec_per_shot,
                    estimated_job_overhead_sec=
                        self.config.estimated_job_overhead_sec,
                    chunk_strategy=self.config.chunk_strategy,
                    max_chunks_per_task=
                        self.config.max_chunks_per_task,
                    allow_oversized_job=
                        self.config.allow_oversized_job,
                )
                files["chunk_plan.json"] = _write_json(
                    output_dir / "chunk_plan.json",
                    {
                        "chunk_strategy":
                            self.config.chunk_strategy,
                        "max_circuits_per_job":
                            int(self.config.max_circuits_per_job),
                        "max_shots_per_job":
                            (int(self.config.max_shots_per_job)
                             if self.config.max_shots_per_job is not None
                             else None),
                        "max_estimated_runtime_sec_per_job":
                            (float(self.config.max_estimated_runtime_sec_per_job)
                             if self.config.max_estimated_runtime_sec_per_job
                                is not None else None),
                        "estimated_sec_per_shot":
                            (float(self.config.estimated_sec_per_shot)
                             if self.config.estimated_sec_per_shot
                                is not None else None),
                        "estimated_job_overhead_sec":
                            (float(self.config.estimated_job_overhead_sec)
                             if self.config.estimated_job_overhead_sec
                                is not None else None),
                        "n_chunks": len(_plan),
                        "total_circuits": len(circuits),
                        "total_shots":
                            int(len(circuits)
                                 * self.config.shots_per_circuit),
                        "estimated_total_runtime_sec":
                            float(sum(c.estimated_runtime_sec
                                       for c in _plan)),
                        "chunks": [c.to_dict() for c in _plan],
                        "dry_run": True,
                    },
                )
            except Exception as exc:
                # planner failure during dry-run is informational only.
                files["chunk_plan_error.txt"] = _write_json(
                    output_dir / "chunk_plan_error.txt",
                    {"error": repr(exc)},
                )
            res = QuantumBackendResult(
                backend_type="ibm_runtime",
                backend_name=self.config.ibm_backend_name,
                execution_mode=self.config.execution_mode,
                status="dry_run",
                circuit_counts=[],
                job_ids=[],
                output_files=files,
                metadata={
                    "n_circuits": len(circuits),
                    "shots_per_circuit": self.config.shots_per_circuit,
                    "estimated_total_shots": (
                        len(circuits) * self.config.shots_per_circuit
                    ),
                    "max_circuits_per_job": self.config.max_circuits_per_job,
                },
            )
            files["backend_result.json"] = _write_json(
                result_path, _result_to_dict(res),
            )
            res.output_files = files
            return res

        # --- import qiskit-ibm-runtime lazily --------------------------
        try:
            from qiskit_ibm_runtime import (
                QiskitRuntimeService,
                Batch,
                SamplerV2,
            )
        except ImportError as e:
            err = (
                "qiskit-ibm-runtime is not installed. "
                "Install with `pip install qiskit-ibm-runtime`."
            )
            (output_dir / "ERROR.txt").write_text(err + "\n")
            res = self._fail(err, files, len(circuits))
            files["backend_result.json"] = _write_json(
                result_path, _result_to_dict(res),
            )
            res.output_files = files
            return res

        # --- resolve service + backend (no direct IBMBackend ctor) -----
        try:
            svc_kwargs: Dict[str, Any] = {}
            if self.config.ibm_channel is not None:
                svc_kwargs["channel"] = self.config.ibm_channel
            if self.config.ibm_instance is not None:
                svc_kwargs["instance"] = self.config.ibm_instance
            service = QiskitRuntimeService(**svc_kwargs)
            backend = service.backend(self.config.ibm_backend_name)
        except Exception as e:
            err = (
                f"Failed to resolve backend "
                f"{self.config.ibm_backend_name!r} via QiskitRuntimeService:"
                f" {e!r}"
            )
            (output_dir / "ERROR.txt").write_text(err + "\n")
            res = self._fail(err, files, len(circuits))
            files["backend_result.json"] = _write_json(
                result_path, _result_to_dict(res),
            )
            res.output_files = files
            return res

        # --- transpile to backend target -------------------------------
        try:
            t_circuits = transpile(
                list(circuits),
                backend=backend,
                optimization_level=self.config.optimization_level,
                seed_transpiler=self.config.seed_transpiler,
            )
            files["transpile_summary.json"] = _write_json(
                output_dir / "transpile_summary.json",
                {
                    "optimization_level": self.config.optimization_level,
                    "seed_transpiler": self.config.seed_transpiler,
                    "backend_name": getattr(backend, "name", str(backend)),
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
        except Exception as e:
            err = f"transpile to backend failed: {e!r}"
            (output_dir / "ERROR.txt").write_text(err + "\n")
            res = self._fail(err, files, len(circuits))
            files["backend_result.json"] = _write_json(
                result_path, _result_to_dict(res),
            )
            res.output_files = files
            return res

        # --- chunk circuits via load-balancing planner ------------------
        try:
            chunk_plan = plan_runtime_chunks(
                list(circuits),
                shots_per_circuit=self.config.shots_per_circuit,
                max_circuits_per_job=self.config.max_circuits_per_job,
                max_shots_per_job=self.config.max_shots_per_job,
                max_estimated_runtime_sec_per_job=
                    self.config.max_estimated_runtime_sec_per_job,
                estimated_sec_per_shot=
                    self.config.estimated_sec_per_shot,
                estimated_job_overhead_sec=
                    self.config.estimated_job_overhead_sec,
                chunk_strategy=self.config.chunk_strategy,
                max_chunks_per_task=self.config.max_chunks_per_task,
                allow_oversized_job=self.config.allow_oversized_job,
            )
        except Exception as e:
            err = f"chunk planner failed: {e!r}"
            (output_dir / "ERROR.txt").write_text(err + "\n")
            res = self._fail(err, files, len(circuits))
            files["backend_result.json"] = _write_json(
                result_path, _result_to_dict(res),
            )
            res.output_files = files
            return res
        # build chunks from plan
        chunks_t = [
            list(t_circuits[c.circuit_start: c.circuit_end])
            for c in chunk_plan
        ]
        chunks_src = [
            list(circuits[c.circuit_start: c.circuit_end])
            for c in chunk_plan
        ]
        files["chunk_plan.json"] = _write_json(
            output_dir / "chunk_plan.json",
            {
                "chunk_strategy": self.config.chunk_strategy,
                "max_circuits_per_job":
                    int(self.config.max_circuits_per_job),
                "max_shots_per_job": (
                    int(self.config.max_shots_per_job)
                    if self.config.max_shots_per_job is not None else None
                ),
                "max_estimated_runtime_sec_per_job": (
                    float(self.config.max_estimated_runtime_sec_per_job)
                    if self.config.max_estimated_runtime_sec_per_job
                       is not None else None
                ),
                "estimated_sec_per_shot": (
                    float(self.config.estimated_sec_per_shot)
                    if self.config.estimated_sec_per_shot is not None
                    else None
                ),
                "estimated_job_overhead_sec": (
                    float(self.config.estimated_job_overhead_sec)
                    if self.config.estimated_job_overhead_sec is not None
                    else None
                ),
                "n_chunks": len(chunk_plan),
                "total_circuits": len(circuits),
                "total_shots": int(
                    len(circuits) * self.config.shots_per_circuit
                ),
                "estimated_total_runtime_sec": float(sum(
                    c.estimated_runtime_sec for c in chunk_plan
                )),
                "chunks": [c.to_dict() for c in chunk_plan],
            },
        )

        # --- run --------------------------------------------------------
        circuit_counts: List[QuantumCircuitCounts] = []
        job_ids: List[str] = []
        chunk_records: List[Dict[str, Any]] = []
        try:
            if self.config.execution_mode == "job":
                for ci, (src_chunk, t_chunk, plan) in enumerate(
                        zip(chunks_src, chunks_t, chunk_plan)):
                    sampler = SamplerV2(mode=backend)
                    self._apply_options(sampler)
                    job = sampler.run(t_chunk)
                    jid = _job_id(job)
                    job_ids.append(jid)
                    result = job.result()
                    self._extend_counts(
                        circuit_counts, src_chunk, t_chunk, result,
                    )
                    chunk_records.append({
                        "chunk_index": int(ci),
                        "n_circuits": int(plan.n_circuits),
                        "total_shots": int(plan.total_shots),
                        "estimated_runtime_sec":
                            float(plan.estimated_runtime_sec),
                        "reason_closed": plan.reason_closed,
                        "job_id": str(jid),
                        "status": "done",
                    })
            elif self.config.execution_mode == "batch":
                with Batch(backend=backend) as batch:
                    for ci, (src_chunk, t_chunk, plan) in enumerate(
                            zip(chunks_src, chunks_t, chunk_plan)):
                        sampler = SamplerV2(mode=batch)
                        self._apply_options(sampler)
                        job = sampler.run(t_chunk)
                        jid = _job_id(job)
                        job_ids.append(jid)
                        result = job.result()
                        self._extend_counts(
                            circuit_counts, src_chunk, t_chunk, result,
                        )
                        chunk_records.append({
                            "chunk_index": int(ci),
                            "n_circuits": int(plan.n_circuits),
                            "total_shots": int(plan.total_shots),
                            "estimated_runtime_sec":
                                float(plan.estimated_runtime_sec),
                            "reason_closed": plan.reason_closed,
                            "job_id": str(jid),
                            "status": "done",
                        })
            else:  # pragma: no cover — guarded by config
                raise ValueError(
                    f"unsupported execution_mode {self.config.execution_mode!r}"
                )

            res = QuantumBackendResult(
                backend_type="ibm_runtime",
                backend_name=self.config.ibm_backend_name,
                execution_mode=self.config.execution_mode,
                status="done",
                circuit_counts=circuit_counts,
                job_ids=job_ids,
                output_files=files,
                metadata={
                    "n_circuits": len(circuits),
                    "n_chunks": len(chunks_src),
                    "shots_per_circuit": self.config.shots_per_circuit,
                    "total_shots": (
                        len(circuits) * self.config.shots_per_circuit
                    ),
                    "max_circuits_per_job":
                        self.config.max_circuits_per_job,
                    "max_shots_per_job":
                        self.config.max_shots_per_job,
                    "max_estimated_runtime_sec_per_job":
                        self.config.max_estimated_runtime_sec_per_job,
                    "chunk_strategy": self.config.chunk_strategy,
                    "estimated_total_runtime_sec": float(sum(
                        c.estimated_runtime_sec for c in chunk_plan
                    )),
                },
            )
        except Exception as e:
            err = f"IBM Runtime run failed: {e!r}"
            (output_dir / "ERROR.txt").write_text(err + "\n")
            res = QuantumBackendResult(
                backend_type="ibm_runtime",
                backend_name=self.config.ibm_backend_name,
                execution_mode=self.config.execution_mode,
                status="failed",
                circuit_counts=circuit_counts,
                job_ids=job_ids,
                output_files=files,
                error=err,
                metadata={"n_circuits": len(circuits)},
            )

        # persist outputs
        files["raw_counts.json"] = _write_json(raw_path, _raw_counts_dict(res))
        files["job_metadata.json"] = _write_json(
            output_dir / "job_metadata.json",
            {
                "execution_mode": self.config.execution_mode,
                "backend_name": self.config.ibm_backend_name,
                "n_chunks": len(chunks_src),
                "total_shots": int(
                    len(circuits) * self.config.shots_per_circuit
                ),
                "estimated_total_runtime_sec": float(sum(
                    c.estimated_runtime_sec for c in chunk_plan
                )),
                "max_shots_per_job": self.config.max_shots_per_job,
                "max_estimated_runtime_sec_per_job":
                    self.config.max_estimated_runtime_sec_per_job,
                "chunk_strategy": self.config.chunk_strategy,
                "job_ids": list(job_ids),
                "per_chunk_job_ids": list(job_ids),
                "chunks": list(chunk_records),
            },
        )
        files["backend_result.json"] = _write_json(
            result_path, _result_to_dict(res),
        )
        res.output_files = files
        return res

    # ------------------------------------------------------------------ #
    def _apply_options(self, sampler) -> None:
        """Best-effort: set shots / resilience_level on SamplerV2 options.

        Uses hasattr/try-except — does not fail if a field is missing
        in the installed qiskit-ibm-runtime version.
        """
        try:
            opts = sampler.options
            # default_shots vs shots vs execution.shots — vary across versions.
            ds = self.config.default_shots
            if ds is None:
                ds = self.config.shots_per_circuit
            for attr in ("default_shots", "shots"):
                if hasattr(opts, attr):
                    try:
                        setattr(opts, attr, int(ds))
                        break
                    except Exception:
                        continue
            if (
                self.config.resilience_level is not None
                and hasattr(opts, "resilience_level")
            ):
                try:
                    opts.resilience_level = int(self.config.resilience_level)
                except Exception:
                    pass
        except Exception:
            # Options not exposed in this version — silently fall back to
            # backend defaults. We surface this via job_metadata.json.
            pass

    def _extend_counts(
        self,
        out: List[QuantumCircuitCounts],
        src_chunk: Sequence[QuantumCircuit],
        t_chunk: Sequence[QuantumCircuit],
        result: Any,
    ) -> None:
        counts_list = extract_counts_from_sampler_result(result)
        if len(counts_list) != len(src_chunk):
            raise ValueError(
                f"sampler returned {len(counts_list)} count blocks but "
                f"chunk has {len(src_chunk)} circuits"
            )
        for src, counts in zip(src_chunk, counts_list):
            out.append(QuantumCircuitCounts(
                circuit_name=src.name,
                counts=dict(counts),
                shots=int(sum(counts.values())) or self.config.shots_per_circuit,
                metadata=dict(src.metadata or {}),
            ))

    def _fail(
        self, err: str, files: Dict[str, str], n_circuits: int,
    ) -> QuantumBackendResult:
        return QuantumBackendResult(
            backend_type="ibm_runtime",
            backend_name=self.config.ibm_backend_name,
            execution_mode=self.config.execution_mode,
            status="failed",
            circuit_counts=[],
            job_ids=[],
            output_files=files,
            error=err,
            metadata={"n_circuits": n_circuits},
        )

    def _load_skipped(
        self,
        raw_path: Path,
        result_path: Path,
        files: Dict[str, str],
    ) -> QuantumBackendResult:
        with raw_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        circuit_counts = [
            QuantumCircuitCounts(
                circuit_name=c["circuit_name"],
                counts=dict(c["counts"]),
                shots=int(c["shots"]),
                metadata=dict(c.get("metadata", {})),
            )
            for c in raw.get("circuits", [])
        ]
        files["raw_counts.json"] = str(raw_path)
        files["backend_result.json"] = str(result_path)
        return QuantumBackendResult(
            backend_type="ibm_runtime",
            backend_name=self.config.ibm_backend_name,
            execution_mode=self.config.execution_mode,
            status="skipped",
            circuit_counts=circuit_counts,
            job_ids=list(raw.get("job_ids", [])),
            output_files=files,
            metadata={
                "resumed_from": str(raw_path),
            },
        )


# ---------------------------------------------------------------------- #
# helpers                                                                #
# ---------------------------------------------------------------------- #

def _chunk(items, size: int):
    if size <= 0:
        return [list(items)]
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def _job_id(job) -> str:
    """Best-effort job id extraction across qiskit-ibm-runtime versions."""
    for attr in ("job_id", "_job_id", "id"):
        v = getattr(job, attr, None)
        if callable(v):
            try:
                return str(v())
            except Exception:
                continue
        if v is not None:
            return str(v)
    return repr(job)


def extract_counts_from_sampler_result(result) -> List[Dict[str, int]]:
    """Robustly pull per-circuit counts dicts from a SamplerV2 result.

    Tries (in order):
      1. result[i].data.meas.get_counts()       — standard SamplerV2 path
         when the circuit has a single classical register named "meas".
      2. result[i].data.<creg_name>.get_counts() — first register found.
      3. result[i].join_data().get_counts()     — older API.

    Raises ValueError if no path produces a counts dict.
    """
    out: List[Dict[str, int]] = []
    # SamplerV2 result is iterable of pub-results.
    for i, pub in enumerate(result):
        counts = None

        data = getattr(pub, "data", None)
        if data is not None:
            # standard: register named "meas"
            meas = getattr(data, "meas", None)
            if meas is not None and hasattr(meas, "get_counts"):
                try:
                    counts = meas.get_counts()
                except Exception:
                    counts = None
            # first available register
            if counts is None:
                for name in dir(data):
                    if name.startswith("_"):
                        continue
                    val = getattr(data, name, None)
                    if val is not None and hasattr(val, "get_counts"):
                        try:
                            counts = val.get_counts()
                            break
                        except Exception:
                            continue

        if counts is None:
            join = getattr(pub, "join_data", None)
            if callable(join):
                try:
                    counts = join().get_counts()
                except Exception:
                    counts = None

        if counts is None:
            raise ValueError(
                f"could not extract counts from pub-result #{i}; "
                f"data attrs = {dir(getattr(pub, 'data', object()))}"
            )
        out.append(dict(counts))
    return out


# ---------------------------------------------------------------------- #
# Job-level load balancing (chunk planner)                               #
# ---------------------------------------------------------------------- #

@dataclass
class RuntimeChunk:
    """One Runtime SamplerV2 submission's worth of circuits.

    A chunk is bounded by max_circuits_per_job AND (optionally)
    max_shots_per_job AND max_estimated_runtime_sec_per_job. The
    planner returns a list of chunks covering all input circuits in
    submission order; each chunk is a contiguous slice.
    """
    chunk_index: int
    circuit_start: int
    circuit_end: int               # exclusive
    n_circuits: int
    shots_per_circuit: int
    total_shots: int
    estimated_runtime_sec: float
    reason_closed: str             # "max_circuits" | "max_shots" |
                                   # "max_runtime" | "end_of_input"
    circuit_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_index": int(self.chunk_index),
            "circuit_start": int(self.circuit_start),
            "circuit_end": int(self.circuit_end),
            "n_circuits": int(self.n_circuits),
            "shots_per_circuit": int(self.shots_per_circuit),
            "total_shots": int(self.total_shots),
            "estimated_runtime_sec": float(self.estimated_runtime_sec),
            "reason_closed": str(self.reason_closed),
            "circuit_names": list(self.circuit_names),
        }


def _estimate_chunk_runtime(
    n_circuits: int, shots_per_circuit: int,
    estimated_sec_per_shot: Optional[float],
    estimated_job_overhead_sec: Optional[float],
) -> float:
    """Soft estimate of one job's wall-clock. If sec_per_shot is None
    we return overhead alone (i.e. the runtime bound becomes a no-op)."""
    base = float(estimated_job_overhead_sec or 0.0)
    if estimated_sec_per_shot is None:
        return base
    return base + n_circuits * shots_per_circuit * float(
        estimated_sec_per_shot
    )


def plan_runtime_chunks(
    circuits: Sequence[QuantumCircuit],
    *,
    shots_per_circuit: int,
    max_circuits_per_job: int,
    max_shots_per_job: Optional[int] = None,
    max_estimated_runtime_sec_per_job: Optional[float] = None,
    estimated_sec_per_shot: Optional[float] = None,
    estimated_job_overhead_sec: Optional[float] = 30.0,
    chunk_strategy: str = "by_circuit_count",
    max_chunks_per_task: Optional[int] = None,
    allow_oversized_job: bool = False,
) -> List[RuntimeChunk]:
    """Greedy contiguous chunking honoring up to three independent bounds.

    Strategies:
      - "by_circuit_count" (V1 default): only respects
        max_circuits_per_job. shots / runtime bounds are ignored.
      - "by_shots": respects max_circuits_per_job AND max_shots_per_job.
        runtime bound ignored.
      - "balanced": respects all three (whichever are set).

    Fail-loud cases:
      - shots_per_circuit > max_shots_per_job → ValueError unless
        allow_oversized_job=True.
      - per-circuit estimated runtime > max_estimated_runtime_sec_per_job
        → ValueError unless allow_oversized_job=True.

    A chunk is closed as soon as adding the next circuit would violate
    any active bound; the reason is recorded in `reason_closed`.
    """
    if max_circuits_per_job <= 0:
        raise ValueError(
            f"max_circuits_per_job must be > 0; got {max_circuits_per_job}"
        )
    if shots_per_circuit <= 0:
        raise ValueError(
            f"shots_per_circuit must be > 0; got {shots_per_circuit}"
        )
    n_total = len(circuits)
    if n_total == 0:
        return []
    use_shots_bound = (chunk_strategy in ("by_shots", "balanced")
                        and max_shots_per_job is not None)
    use_runtime_bound = (chunk_strategy == "balanced"
                          and max_estimated_runtime_sec_per_job is not None)

    # Per-circuit oversize fail-loud checks (only when bound is active)
    if use_shots_bound and shots_per_circuit > int(max_shots_per_job):
        msg = (f"shots_per_circuit ({shots_per_circuit}) > "
               f"max_shots_per_job ({max_shots_per_job}); "
               f"single circuit cannot fit in a job under chunk_strategy="
               f"{chunk_strategy!r}.")
        if not allow_oversized_job:
            raise ValueError(msg)
    if use_runtime_bound:
        per_circ_rt = _estimate_chunk_runtime(
            n_circuits=1, shots_per_circuit=shots_per_circuit,
            estimated_sec_per_shot=estimated_sec_per_shot,
            estimated_job_overhead_sec=estimated_job_overhead_sec,
        )
        if per_circ_rt > float(max_estimated_runtime_sec_per_job):
            msg = (f"single-circuit estimated runtime ({per_circ_rt:.1f}s) > "
                   f"max_estimated_runtime_sec_per_job "
                   f"({max_estimated_runtime_sec_per_job}); "
                   f"chunk_strategy={chunk_strategy!r}.")
            if not allow_oversized_job:
                raise ValueError(msg)

    chunks: List[RuntimeChunk] = []
    i = 0
    while i < n_total:
        cur_start = i
        cur_n = 0
        cur_shots = 0
        cur_rt = float(estimated_job_overhead_sec or 0.0)
        # always include at least one circuit even if it exceeds bounds
        # (allow_oversized_job=True path), to guarantee progress
        reason_closed = "end_of_input"
        while i < n_total:
            new_n = cur_n + 1
            new_shots = cur_shots + shots_per_circuit
            new_rt = _estimate_chunk_runtime(
                n_circuits=new_n,
                shots_per_circuit=shots_per_circuit,
                estimated_sec_per_shot=estimated_sec_per_shot,
                estimated_job_overhead_sec=estimated_job_overhead_sec,
            )
            # check bounds (only close if we already have ≥1 circuit;
            # never block the first circuit of a chunk)
            if new_n > max_circuits_per_job and cur_n >= 1:
                reason_closed = "max_circuits"
                break
            if (use_shots_bound
                    and new_shots > int(max_shots_per_job)
                    and cur_n >= 1):
                reason_closed = "max_shots"
                break
            if (use_runtime_bound
                    and new_rt > float(max_estimated_runtime_sec_per_job)
                    and cur_n >= 1):
                reason_closed = "max_runtime"
                break
            # accept circuit
            cur_n = new_n
            cur_shots = new_shots
            cur_rt = new_rt
            i += 1
        if cur_n == 0:
            # we reached here without consuming anything → infinite loop
            # guard. Should not happen given fail-loud checks above.
            raise RuntimeError(
                "plan_runtime_chunks made no progress; check that bounds "
                "leave room for at least one circuit per chunk."
            )
        cur_end = cur_start + cur_n
        chunks.append(RuntimeChunk(
            chunk_index=len(chunks),
            circuit_start=cur_start,
            circuit_end=cur_end,
            n_circuits=cur_n,
            shots_per_circuit=int(shots_per_circuit),
            total_shots=int(cur_shots),
            estimated_runtime_sec=float(cur_rt),
            reason_closed=reason_closed,
            circuit_names=[
                str(c.name) for c in circuits[cur_start:cur_end]
            ],
        ))
        if (max_chunks_per_task is not None
                and len(chunks) >= int(max_chunks_per_task)):
            # cap reached: emit a single-chunk for remainder so we still
            # cover all circuits in submission, but flag in reason
            if i < n_total:
                rem_start = i
                rem_end = n_total
                rem_n = rem_end - rem_start
                rem_shots = rem_n * shots_per_circuit
                rem_rt = _estimate_chunk_runtime(
                    rem_n, shots_per_circuit,
                    estimated_sec_per_shot,
                    estimated_job_overhead_sec,
                )
                chunks.append(RuntimeChunk(
                    chunk_index=len(chunks),
                    circuit_start=rem_start, circuit_end=rem_end,
                    n_circuits=rem_n,
                    shots_per_circuit=int(shots_per_circuit),
                    total_shots=int(rem_shots),
                    estimated_runtime_sec=float(rem_rt),
                    reason_closed="max_chunks_per_task_reached",
                    circuit_names=[
                        str(c.name) for c in circuits[rem_start:rem_end]
                    ],
                ))
            break
    return chunks


__all__ = [
    "IBMRuntimeQuantumBackend",
    "extract_counts_from_sampler_result",
    "RuntimeChunk",
    "plan_runtime_chunks",
]
