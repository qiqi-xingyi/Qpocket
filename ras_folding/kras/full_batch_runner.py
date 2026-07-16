# Author: Yuqi Zhang
"""KrasFullBatchRunner — end-to-end per-task pipeline driver.

This is the single, only pipeline flow. There is no ``sampler_mode``
switch and no legacy front end: every task runs the same seven-stage
pipeline (see ``doc/SYSTEM_REPORT.md``):

    A. V2 environment+corridor prior  (build_environment_prior /
       build_corridor_prior)            ← env-conditioned, WIRED here
    B. closed-form HEA θ                (MomentMatchInitializer)
    C. HEA circuit assembly            (ansatz="hea_with_tf")
    D. quantum sampling                (QuantumBackendBaseSampler; AER/IBM)
    E. classical imaginary-time reject (QuantumImaginaryTimeSampler)
    F. Hybrid SQD refinement           (SubspaceDiagonalizationRefiner,
                                        coupling_mode="hybrid")
    G. postprocess + landscape + analysis  (downstream docking/validation
       live in run_sampling.py / run_oracle_docking_eval.py, both
       orchestrated by run_pipeline.py)

Stage A is fully wired: the moment matcher samples env-consistent
bitstrings (``_sample_v2_full``) from the V2 corridor+environment prior,
falling back to the env-blind random sampler only if the reference PDB /
environment cannot be built.

This module is GLUE. It does NOT introduce new sampling / scoring /
clustering algorithms; everything is composed from the existing modules.

Resume: a `<task_dir>/DONE` file marks completion. With overwrite=False,
done tasks are skipped. Failed tasks write `<task_dir>/ERROR.txt` and
are recorded in the global summary.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import statistics
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ras_folding.densify.dense_filler import (
    PerturbationDenseFiller,
    write_dense_outputs,
)
from ras_folding.kras.landscape import LandscapeReconstructor
from ras_folding.kras.report import write_global_report, write_task_report
from ras_folding.kras.structure_analysis import StructureAnalyzer
from ras_folding.kras.task_loader import KrasTask
from ras_folding.postprocess.prediction_postprocessor import (
    PredictionPostProcessor,
)
from ras_folding.quantum.backend_config import QuantumBackendConfig
from ras_folding.quantum.circuit_builder import QuantumCircuitBuilder
from ras_folding.refinement.subspace_diagonalization import (
    SubspaceDiagonalizationRefiner,
)
from ras_folding.sampler.context import SamplingContext
from ras_folding.sampler.filter_hamiltonian import FilterHamiltonian
from ras_folding.sampler.imaginary_time_sampler import (
    QuantumImaginaryTimeSampler,
)
from ras_folding.sampler.quantum_base_sampler import QuantumBackendBaseSampler
from ras_folding.sampler.sample_types import CandidateSample, SampleBatch
from ras_folding.scoring.full_energy import FullEnergyScorer
from ras_folding.scoring.mj_contact import load_mj_table_default


# ---------------------------------------------------------------------- #
# small helpers                                                          #
# ---------------------------------------------------------------------- #

def _is_dense(sample) -> bool:
    return bool(
        (sample.metadata.get("densify") or {}).get("is_perturbed", False)
    )


def _coerce_json(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _coerce_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_coerce_json(x) for x in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return o


def _favorable_contact_miss_mean(samples: Sequence[CandidateSample]):
    vals = [
        float(s.filter_terms.get("favorable_contact_miss", 0.0))
        for s in samples
        if s.valid and s.filter_terms
    ]
    if not vals:
        return None
    return float(statistics.fmean(vals))


def _per_tau_metrics(b: SampleBatch) -> Dict[str, Any]:
    rep = b.report()
    s = rep["summary"]
    invalid_counts = s.get("invalid_reason_counts", {}) or {}
    return {
        "tau": rep["tau"],
        "n_raw": rep["n_raw"],
        "n_valid": rep["n_valid"],
        "n_invalid": rep["n_invalid"],
        "n_accepted": rep["n_accepted"],
        "valid_rate": rep["valid_rate"],
        "acceptance_rate": rep["acceptance_rate"],
        "fallback_triggered": int(invalid_counts.get("fallback_triggered", 0)),
        "long_range_clash": int(invalid_counts.get("long_range_clash", 0)),
        "mean_filter_energy_accepted": s.get("mean_filter_energy_accepted"),
        "mean_full_energy_accepted": s.get("mean_full_energy_accepted"),
        "favorable_contact_miss_mean": _favorable_contact_miss_mean(b.samples),
    }


def _pool_eligible(batches) -> List[CandidateSample]:
    seen: Dict[str, CandidateSample] = {}
    for b in batches:
        for s in b.samples:
            if not s.is_eligible_for_refinement():
                continue
            key = (
                s.bitstring if s.bitstring is not None else f"id_{id(s)}"
            )
            if key in seen:
                seen[key].count += s.count
            else:
                seen[key] = s
    return list(seen.values())


# ---------------------------------------------------------------------- #
# runner                                                                 #
# ---------------------------------------------------------------------- #

@dataclass
class _RunnerDefaults:
    """Per-task pipeline knobs not in the QuantumBackendConfig itself."""
    taus: Tuple[float, ...] = (0.0, 0.1, 0.2)
    n_circuits: int = 4
    seed: int = 2024

    densify_top_parents: int = 50
    densify_children_per_parent: int = 8
    densify_angular_sigmas_deg: Tuple[float, ...] = (1.5, 3.0, 5.0)
    densify_max_local_rmsd: float = 1.0
    densify_energy_window: float = 10.0
    densify_perturbation_mass: float = 0.3

    refiner_max_subspace_size: int = 200
    refiner_k_neighbors: int = 10
    refiner_kappa: float = 0.2
    refiner_n_modes: int = 5
    refiner_max_dense_fraction: float = 0.6
    refiner_min_original_fraction: float = 0.3

    post_dedup_rmsd_threshold: float = 0.5
    post_basin_rmsd_threshold: float = 1.5
    post_top_k_candidates: int = 20
    post_top_k_basins: int = 5

    # ------------------------------------------------------------------
    # the ONLY front end in full_pipline
    # ------------------------------------------------------------------
    # The sampler is fixed to HEA with V2-derived θ from closed-form
    # moment matching (genuine entanglement). There is no runtime switch
    # and no legacy front end; the "hea_moment_matched" provenance label
    # is stamped into the per-task payload at run time.

    # Moment-match initializer settings (Stage A/B)
    moment_match_K: int = 500           # V2 prior samples for statistics
    moment_match_reps: int = 1           # HEA layers (reps=1 supported)

    # V2 environment+corridor prior (Stage A) — wired into the
    # moment matcher so it draws env-consistent bitstrings. These mirror
    # the V2 batch runner's de-leaked defaults.
    prior_crystal_leakage_mode: str = "perturbed"
    prior_perturbation_sigma: float = 5.0
    prior_perturbation_clip: float = 2.0

    # Refinement coupling for Hybrid SQD (Stage F). full_pipline is
    # single-flow: the sole coupling is "hybrid" = α·T_Pauli + β·T_RMSD
    # (the exact Pauli Hamming-1 matrix element composed with the
    # RMSD-Gaussian kernel). Recorded for provenance; not a runtime switch.
    refiner_coupling_mode: str = "hybrid"
    refiner_g_quantum: float = 0.03      # transverse-field coupling
    refiner_alpha_pauli: float = 0.5
    refiner_alpha_rmsd: float = 0.5


class KrasFullBatchRunner:
    """End-to-end per-task driver, with resume + per-task isolation."""

    def __init__(
        self,
        backend_config: QuantumBackendConfig,
        circuit_builder_config: Optional[Dict[str, Any]] = None,
        pdb_dir: Optional[Path] = None,
        output_root: Path = Path("logs/kras_full_batch"),
        run_name: str = "default_run",
        overwrite: bool = False,
        fail_fast: bool = False,
        densify_enabled: bool = True,
        postprocess_enabled: bool = True,
        landscape_enabled: bool = True,
        max_tasks: Optional[int] = None,
        defaults: Optional[_RunnerDefaults] = None,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.backend_config = backend_config
        self.circuit_builder_config = dict(circuit_builder_config or {})
        self.pdb_dir = Path(pdb_dir) if pdb_dir is not None else None
        self.output_root = Path(output_root) / run_name
        self.run_name = run_name
        self.overwrite = bool(overwrite)
        self.fail_fast = bool(fail_fast)
        self.densify_enabled = bool(densify_enabled)
        self.postprocess_enabled = bool(postprocess_enabled)
        self.landscape_enabled = bool(landscape_enabled)
        self.max_tasks = (
            None if max_tasks is None else int(max_tasks)
        )
        self.defaults = defaults or _RunnerDefaults()
        self.progress_callback = progress_callback

    # ------------------------------------------------------------------ #
    def run(
        self,
        tasks: List[KrasTask],
        schema_summary: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        self.output_root.mkdir(parents=True, exist_ok=True)
        if schema_summary is not None:
            (self.output_root / "tasks_schema.json").write_text(
                json.dumps(_coerce_json(schema_summary), indent=2),
                encoding="utf-8",
            )

        results: List[Dict[str, Any]] = []
        n_runnable = (
            len(tasks) if self.max_tasks is None
            else min(len(tasks), self.max_tasks)
        )
        for i, task in enumerate(tasks[:n_runnable]):
            task_dir = self.output_root / task.task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            done_marker = task_dir / "DONE"

            if done_marker.exists() and not self.overwrite:
                # resume: load saved per-task summary if present
                summary_path = task_dir / "task_summary.json"
                if summary_path.exists():
                    payload = json.loads(summary_path.read_text())
                else:
                    payload = {"task": {"task_id": task.task_id}, "skipped": True}
                payload["__skipped"] = True
                results.append(payload)
                self._notify("task_skipped", {
                    "i": i, "task_id": task.task_id, "task_dir": str(task_dir),
                })
                continue

            self._notify("task_start", {
                "i": i, "task_id": task.task_id, "task_dir": str(task_dir),
            })
            t0 = time.time()
            try:
                payload = self._run_single_task(task, task_dir)
                payload["__elapsed_sec"] = time.time() - t0
                # write per-task summary.json
                (task_dir / "task_summary.json").write_text(
                    json.dumps(_coerce_json(payload), indent=2),
                    encoding="utf-8",
                )
                # markdown report
                try:
                    write_task_report(task_dir, payload)
                except Exception as e:
                    payload.setdefault("warnings", []).append(
                        f"task_report_failed: {e!r}"
                    )
                done_marker.write_text("done\n")
                results.append(payload)
                self._notify("task_done", {
                    "i": i, "task_id": task.task_id, "task_dir": str(task_dir),
                    "elapsed_sec": payload["__elapsed_sec"],
                })
            except Exception as e:
                err = traceback.format_exc()
                (task_dir / "ERROR.txt").write_text(err)
                payload = {
                    "task": {"task_id": task.task_id},
                    "error": repr(e),
                    "__failed": True,
                    "__elapsed_sec": time.time() - t0,
                }
                results.append(payload)
                self._notify("task_failed", {
                    "i": i, "task_id": task.task_id,
                    "error": repr(e), "task_dir": str(task_dir),
                })
                if self.fail_fast:
                    break

        # global summaries
        self._write_global_summary(results)
        return results

    # ================================================================== #
    # internals                                                          #
    # ================================================================== #

    def _notify(self, event: str, payload: Dict[str, Any]) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(event, payload)
        except Exception:
            pass  # never fail on progress callback errors

    # ------------------------------------------------------------------ #
    def _run_single_task(
        self, task: KrasTask, task_dir: Path,
    ) -> Dict[str, Any]:
        d = self.defaults
        ctx = SamplingContext(
            encoder_inputs=task.encoder_inputs,
            sequence=task.sequence,
            metadata={"case_id": task.task_id, **task.metadata},
        )
        n_qubits = ctx.encoder_inputs.n_bonds * 6

        # write input.json
        (task_dir / "input.json").write_text(
            json.dumps(_coerce_json({
                "task_id": task.task_id,
                "sequence": task.sequence,
                "n_residues": ctx.encoder_inputs.n_residues,
                "n_bonds": ctx.encoder_inputs.n_bonds,
                "n_qubits": n_qubits,
                "metadata": task.metadata,
                "has_reference_coords": task.reference_coords is not None,
            }), indent=2),
            encoding="utf-8",
        )

        warnings: List[str] = []

        # --- scoring stack (need fh early for moment-match initializer) -
        mj = load_mj_table_default()
        fh = FilterHamiltonian(residue_contact_weights=mj)
        scorer = FullEnergyScorer(
            residue_contact_weights=mj,
            rg_target=None,
            term_weights={
                "overlap_full": 10.0, "contact_full": 1.0,
                "rg": 0.0, "anchor": 1.0, "turn": 0.0,
            },
        )

        # --- Stage A+B: env-conditioned moment matching ---------
        # full_pipline is single-flow. There is no sampler_mode switch:
        # every task derives HEA θ via closed-form moment matching of the
        # V2 corridor+environment prior, then prepares the genuinely
        # entangled hea_with_tf ansatz.
        builder_cfg = dict(self.circuit_builder_config)
        builder_cfg.setdefault("seed", d.seed)
        sampler_mode = "hea_moment_matched"             # fixed (provenance)

        from ras_folding.quantum.moment_match_initializer import (
            MomentMatchConfig, MomentMatchInitializer,
        )

        # Stage A.0: build the V2 environment + corridor prior so the
        # moment matcher samples env-consistent bitstrings
        # (_sample_v2_full) rather than the env-blind random fallback.
        # Any failure (missing PDB, empty environment, no ligand) degrades
        # gracefully to env_ctx=None → fallback_random; the task never
        # aborts here.
        env_ctx, corridor_ctx, prior_payload = self._build_v2_prior(
            task, task_dir,
        )

        mm_cfg = MomentMatchConfig(
            K_samples=d.moment_match_K,
            reps=d.moment_match_reps,
            seed=d.seed,
            use_v2_prior=True,
        )
        mm_initializer = MomentMatchInitializer(
            sampling_context=ctx,
            filter_hamiltonian=fh,
            config=mm_cfg,
            env_ctx=env_ctx,
            corridor_ctx=corridor_ctx,
        )
        mm_result = mm_initializer.compute_theta()
        (task_dir / "moment_match").mkdir(parents=True, exist_ok=True)
        (task_dir / "moment_match" / "summary.json").write_text(
            json.dumps(_coerce_json({
                "sampling_mode": mm_result.sampling_mode,
                "n_valid_samples": mm_result.n_valid_samples,
                "diagnostics": mm_result.diagnostics,
                "z_star_h_filter": mm_result.z_star_h_filter,
                "n_cx_edges": len(mm_result.cx_edges),
                "env_prior": prior_payload,
            }), indent=2), encoding="utf-8",
        )
        moment_match_payload = {
            "sampling_mode": mm_result.sampling_mode,
            "n_valid_samples": mm_result.n_valid_samples,
            "z_star_h_filter": mm_result.z_star_h_filter,
            "env_prior_wired": bool(
                env_ctx is not None and corridor_ctx is not None
            ),
            "env_prior": prior_payload,
            "diagnostics": mm_result.diagnostics,
        }
        builder_cfg["ansatz"] = "hea_with_tf"
        builder_cfg["ansatz_params"] = mm_result.theta
        builder_cfg["reps_hea"] = d.moment_match_reps

        circuit_builder = QuantumCircuitBuilder(**builder_cfg)

        # --- quantum base sampler --------------------------------------
        quantum_dir = task_dir / "quantum"
        qbase = QuantumBackendBaseSampler(
            backend_config=self.backend_config,
            circuit_builder=circuit_builder,
            n_circuits=d.n_circuits,
            seeds=[d.seed + k for k in range(d.n_circuits)],
            output_dir=quantum_dir,
            task_id=task.task_id,
        )

        # (scoring stack already constructed before circuit_builder above)

        # --- imaginary-time sampler ------------------------------------
        its = QuantumImaginaryTimeSampler(
            taus=d.taus,
            shots_per_tau=d.n_circuits * self.backend_config.shots_per_circuit,
            base_sampler=qbase,
            filter_hamiltonian=fh,
            scorer=scorer,
            seed=d.seed,
        )

        # --- step 1: run quantum sampling ------------------------------
        batches = its.sample(ctx)
        per_tau = [_per_tau_metrics(b) for b in batches]

        # detect dry-run / failed status from the most recent quantum run
        # (the AER backend writes one file per call; for IBM runs job_ids
        # are surfaced via batch summary metadata).
        backend_status, backend_job_ids = _read_backend_status(quantum_dir)

        # If dry_run: stop here (no decoded samples are produced because
        # counts_to_candidate_samples was given an empty result).
        # In that case batches[*].n_raw == 0 so the rest is a no-op.
        sampler_dir = task_dir / "sampler"
        sampler_dir.mkdir(exist_ok=True)
        _write_per_tau_csv(sampler_dir / "batch_summary_by_tau.csv", per_tau)

        accepted_pool = _pool_eligible(batches)
        _write_candidate_csv(
            sampler_dir / "accepted_candidates.csv",
            accepted_pool,
            include_full_energy=True,
        )
        _write_candidate_csv(
            sampler_dir / "full_scored_candidates.csv",
            [s for s in accepted_pool if s.full_energy is not None],
            include_full_energy=True,
        )

        # --- step 2: optional densify ----------------------------------
        densify_summary: Optional[Dict[str, Any]] = None
        if self.densify_enabled and accepted_pool:
            filler = PerturbationDenseFiller(
                top_parents=d.densify_top_parents,
                children_per_parent=d.densify_children_per_parent,
                angular_sigmas_deg=d.densify_angular_sigmas_deg,
                max_local_rmsd=d.densify_max_local_rmsd,
                energy_window=d.densify_energy_window,
                perturbation_mass=d.densify_perturbation_mass,
                seed=d.seed,
            )
            dense_res = filler.densify(accepted_pool, scorer, ctx)
            write_dense_outputs(dense_res, task_dir / "densify")
            densify_summary = dense_res.summary
            cands_for_refine = dense_res.all_candidates
        else:
            cands_for_refine = accepted_pool

        # --- step 3: refinement ----------------------------------------
        refinement_dir = task_dir / "refinement"
        refinement_dir.mkdir(exist_ok=True)
        refinement_summary: Dict[str, Any] = {}
        refined_candidates_obj = None
        if cands_for_refine:
            final_batch = SampleBatch(
                samples=cands_for_refine,
                tau=None,
                n_raw=len(cands_for_refine),
                n_valid=sum(1 for s in cands_for_refine if s.valid),
                n_accepted=sum(1 for s in cands_for_refine if s.accepted),
                summary={
                    "merged_from_densify": (
                        self.densify_enabled and densify_summary is not None
                    ),
                },
            )
            refiner = SubspaceDiagonalizationRefiner(
                max_subspace_size=d.refiner_max_subspace_size,
                k_neighbors=d.refiner_k_neighbors,
                kappa=d.refiner_kappa,
                n_modes=d.refiner_n_modes,
                max_dense_fraction=d.refiner_max_dense_fraction,
                min_original_fraction=d.refiner_min_original_fraction,
                # Hybrid SQD (Pauli + RMSD) by default
                coupling_mode=d.refiner_coupling_mode,
                g_quantum=d.refiner_g_quantum,
                alpha_pauli=d.refiner_alpha_pauli,
                alpha_rmsd=d.refiner_alpha_rmsd,
                n_qubits=n_qubits,
            )
            refinement_result = refiner.refine(final_batch)
            refined_candidates_obj = refinement_result
            refinement_summary = dict(refinement_result.summary)
            (refinement_dir / "refinement_summary.json").write_text(
                json.dumps(_coerce_json(refinement_summary), indent=2),
                encoding="utf-8",
            )
            _write_refined_csv(
                refinement_dir / "refined_candidates.csv",
                refinement_result.candidates,
            )

        # --- candidate coords archive (oracle eval consumes this) ----
        # Save coords for accepted_pool + dense children + refined
        # candidates. Sources may overlap — oracle reader can dedup.
        # Failure here MUST NOT break the task; oracle just falls back.
        try:
            _write_candidates_coords_archive(
                task_dir=task_dir,
                accepted_pool=accepted_pool,
                dense_candidates=(
                    densify_summary
                    and getattr(dense_res, "dense_candidates", [])
                ) or [],
                refined_candidates=(
                    refined_candidates_obj.candidates
                    if refined_candidates_obj is not None else []
                ),
            )
        except Exception as e:
            warnings.append(f"candidates_coords_archive_failed: {e!r}")

        # --- step 4: postprocess --------------------------------------
        post_summary: Dict[str, Any] = {}
        post_files: Dict[str, str] = {}
        prediction = None
        if self.postprocess_enabled and refined_candidates_obj is not None:
            post = PredictionPostProcessor(
                dedup_rmsd_threshold=d.post_dedup_rmsd_threshold,
                basin_rmsd_threshold=d.post_basin_rmsd_threshold,
                top_k_candidates=d.post_top_k_candidates,
                top_k_basins=d.post_top_k_basins,
                export_pdb=True,
            )
            prediction = post.process(
                refined_candidates_obj,
                output_dir=task_dir,
                sequence=task.sequence,
            )
            post_summary = prediction.summary
            post_files = prediction.output_files

        # --- step 5: landscape ----------------------------------------
        landscape_summary: Dict[str, Any] = {}
        if (
            self.landscape_enabled
            and refined_candidates_obj is not None
            and refined_candidates_obj.candidates
        ):
            recon = LandscapeReconstructor(
                grid_size=80, plot_enabled=True,
            )
            l_res = recon.reconstruct(
                refined_candidates_obj.candidates,
                output_dir=task_dir / "landscape",
                reference_coords=task.reference_coords,
            )
            landscape_summary = l_res.summary

        # --- step 6: structure analysis -------------------------------
        structure_summary: Dict[str, Any] = {}
        if prediction is not None:
            analyzer = StructureAnalyzer()
            structure_summary = analyzer.analyze(
                prediction,
                reference_coords=task.reference_coords,
                output_dir=task_dir / "analysis",
            )

        # --- assemble payload -----------------------------------------
        payload: Dict[str, Any] = {
            "task": {
                "task_id": task.task_id,
                "sequence": task.sequence,
                "n_residues": ctx.encoder_inputs.n_residues,
                "n_bonds": ctx.encoder_inputs.n_bonds,
                "n_qubits": n_qubits,
                "ref_pdb": task.metadata.get("ref_pdb"),
                "chain_id": task.metadata.get("chain_id"),
                "start_resi": task.metadata.get("start_resi"),
                "end_resi": task.metadata.get("end_resi"),
                "has_native_ref": task.metadata.get("has_native_ref"),
                # CSV-extra grouping fields (None when absent).
                "mutation_group": task.metadata.get("mutation_group"),
                "ligand_family": task.metadata.get("ligand_family"),
                "pocket_module": task.metadata.get("pocket_module"),
                "analysis_role": task.metadata.get("analysis_role"),
                "csv_row": task.metadata.get("csv_row") or {},
            },
            "backend": {
                "backend_type": self.backend_config.backend_type,
                "ibm_backend_name": self.backend_config.ibm_backend_name,
                "execution_mode": self.backend_config.execution_mode,
                "sampler_mode": sampler_mode,
                "ansatz": builder_cfg["ansatz"],
                "n_circuits": d.n_circuits,
                "shots_per_circuit": self.backend_config.shots_per_circuit,
                "total_shots": d.n_circuits * self.backend_config.shots_per_circuit
                                * len(d.taus),
                "paired_tau_sampling": self.backend_config.paired_tau_sampling,
                "dry_run": self.backend_config.dry_run,
                "status": backend_status,
                "job_ids": list(backend_job_ids or []),
            },
            "per_tau": per_tau,
            "moment_match": moment_match_payload,
            "densify": densify_summary,
            "refinement": refinement_summary,
            "post": post_summary,
            "landscape": landscape_summary,
            "structure": structure_summary,
            "top1_pdb": post_files.get("top1_ca.pdb"),
            "basin_pdbs": [
                v for k, v in post_files.items()
                if k.startswith("top_basins/")
            ],
            "warnings": warnings,
        }
        return payload

    # ------------------------------------------------------------------ #
    def _build_v2_prior(
        self, task: KrasTask, task_dir: Path,
    ) -> Tuple[Optional[Any], Optional[Any], Dict[str, Any]]:
        """Build the V2 environment + corridor prior for Stage A.

        Returns ``(env_ctx, corridor_ctx, payload)``. On ANY failure
        (no ``pdb_dir`` configured, reference PDB missing, empty
        environment, no ligand, ...) returns ``(None, None, payload)`` so
        the moment matcher falls back to its env-blind random sampler.
        This method never raises — Stage A degradation must not abort a
        task.
        """
        from ras_folding.prior.environment import build_environment_prior
        from ras_folding.prior.corridor import build_corridor_prior

        d = self.defaults
        meta = task.metadata
        payload: Dict[str, Any] = {"wired": False, "reason": None}

        if self.pdb_dir is None:
            payload["reason"] = "runner has no pdb_dir configured"
            return None, None, payload
        ref_pdb = meta.get("ref_pdb")
        if not ref_pdb:
            payload["reason"] = "task metadata missing ref_pdb"
            return None, None, payload
        pdb_path = Path(self.pdb_dir) / ref_pdb
        if not pdb_path.is_file():
            payload["reason"] = f"reference PDB not found: {pdb_path}"
            return None, None, payload

        try:
            centroid = (
                np.asarray(task.reference_coords, dtype=np.float64).mean(axis=0)
                if task.reference_coords is not None else None
            )
            env_ctx = build_environment_prior(
                pdb_path=str(pdb_path),
                chain_id=meta["chain_id"],
                start_resi=int(meta["start_resi"]),
                end_resi=int(meta["end_resi"]),
                ligand_resname=meta.get("ligand_resname"),
                fragment_ca_centroid=centroid,
            )
            corridor_ctx = build_corridor_prior(
                task.encoder_inputs, env_ctx,
                crystal_leakage_mode=d.prior_crystal_leakage_mode,
                perturbation_sigma=d.prior_perturbation_sigma,
                perturbation_clip_n_sigma=d.prior_perturbation_clip,
                task_id=task.task_id,
            )
        except Exception as e:
            payload["reason"] = f"prior build failed: {e!r}"
            return None, None, payload

        payload.update({
            "wired": True,
            "reason": None,
            "n_env_atoms": int(getattr(env_ctx, "n_env_atoms", 0)),
            "n_ligand_atoms": int(getattr(env_ctx, "n_ligand_atoms", 0)),
            "ligand_selection_mode": getattr(
                env_ctx, "ligand_selection_mode", None,
            ),
            "corridor_mode": getattr(corridor_ctx, "corridor_mode", None),
            "crystal_leakage_mode": d.prior_crystal_leakage_mode,
        })
        try:
            (task_dir / "prior").mkdir(parents=True, exist_ok=True)
            (task_dir / "prior" / "env_corridor_summary.json").write_text(
                json.dumps(_coerce_json(payload), indent=2), encoding="utf-8",
            )
        except Exception:
            pass
        return env_ctx, corridor_ctx, payload

    # ------------------------------------------------------------------ #
    def _write_global_summary(
        self, results: List[Dict[str, Any]],
    ) -> None:
        rows: List[Dict[str, Any]] = []
        successful: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []
        for r in results:
            tid = r.get("task", {}).get("task_id") or "<unknown>"
            if r.get("__failed"):
                failed.append({"task_id": tid, "error": r.get("error")})
                rows.append({
                    "task_id": tid, "status": "failed",
                    "error": r.get("error"),
                })
                continue
            if r.get("__skipped"):
                rows.append({"task_id": tid, "status": "skipped"})
                continue
            successful.append(r)
            post = r.get("post") or {}
            structure = r.get("structure") or {}
            t1 = (structure.get("top1") or {}) if structure else {}
            tk = (structure.get("top_k") or {}) if structure else {}
            backend = r.get("backend") or {}
            ref = r.get("refinement") or {}
            land = r.get("landscape") or {}
            funnel = (land.get("funnel") if land else None) or {}
            spread = (land.get("spread") if land else None) or {}
            task_blk = r.get("task", {}) or {}
            row = {
                "task_id": tid,
                "status": "ok",
                "elapsed_sec": r.get("__elapsed_sec"),
                "n_residues": task_blk.get("n_residues"),
                "n_qubits": task_blk.get("n_qubits"),
                # CSV-extra grouping fields (None when absent in CSV).
                "mutation_group": task_blk.get("mutation_group"),
                "ligand_family": task_blk.get("ligand_family"),
                "pocket_module": task_blk.get("pocket_module"),
                "analysis_role": task_blk.get("analysis_role"),
                "backend_type": backend.get("backend_type"),
                "backend_name": backend.get("ibm_backend_name"),
                "execution_mode": backend.get("execution_mode"),
                "total_shots": backend.get("total_shots"),
                "job_ids": ";".join(backend.get("job_ids") or []),
                "n_after_validity_filter": post.get("n_after_validity_filter"),
                "n_after_structure_dedup": post.get("n_after_structure_dedup"),
                "n_basins": post.get("n_basins"),
                "top_basin_weight": post.get("top_basin_weight"),
                "top1_full_energy": post.get("top1_full_energy"),
                "top1_refined_score": post.get("top1_refined_score"),
                "top1_is_dense": post.get("top1_is_dense"),
                "n_dense_candidates": ref.get("n_dense_candidates"),
                "dense_fraction_in_subspace": ref.get(
                    "dense_fraction_in_subspace"
                ),
                "best_rmsd": funnel.get("best_rmsd"),
                "top1_rmsd": funnel.get("top1_rmsd"),
                "top5_best_rmsd": funnel.get("top5_best_rmsd"),
                "near_native_count_2A": funnel.get("near_native_count_2A"),
                "energy_spread": spread.get("energy_spread"),
                "refined_weight_entropy": (
                    spread.get("refined_weight_entropy")
                    if spread else funnel.get("refined_weight_entropy")
                ),
            }
            rows.append(row)

        # CSV
        csv_path = self.output_root / "global_summary.csv"
        if rows:
            cols = sorted({k for r in rows for k in r.keys()})
            cols = [
                "task_id", "status", "elapsed_sec",
                "n_residues", "n_qubits",
                "mutation_group", "ligand_family",
                "pocket_module", "analysis_role",
                "backend_type", "backend_name", "execution_mode",
                "total_shots", "job_ids",
                "n_after_validity_filter", "n_after_structure_dedup",
                "n_basins", "top_basin_weight",
                "top1_full_energy", "top1_refined_score", "top1_is_dense",
                "n_dense_candidates", "dense_fraction_in_subspace",
                "best_rmsd", "top1_rmsd", "top5_best_rmsd",
                "near_native_count_2A",
                "energy_spread", "refined_weight_entropy",
                "error",
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                for r in rows:
                    w.writerow({k: _csv_val(r.get(k)) for k in cols})

        # JSON
        json_path = self.output_root / "global_summary.json"
        json_path.write_text(
            json.dumps(_coerce_json(rows), indent=2),
            encoding="utf-8",
        )

        # failed_tasks.csv
        if failed:
            with (self.output_root / "failed_tasks.csv").open(
                "w", newline="", encoding="utf-8",
            ) as fh:
                w = csv.DictWriter(fh, fieldnames=["task_id", "error"])
                w.writeheader()
                for f in failed:
                    w.writerow({"task_id": f["task_id"], "error": f["error"]})

        # global_report.md (compact distributions)
        global_payload = self._build_global_payload(successful, failed)
        write_global_report(self.output_root, global_payload)

    # ------------------------------------------------------------------ #
    def _build_global_payload(
        self,
        successful: List[Dict[str, Any]],
        failed: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        n_total = len(successful) + len(failed)
        total_shots = sum(
            r.get("backend", {}).get("total_shots") or 0
            for r in successful
        )
        total_jobs = sum(
            len(r.get("backend", {}).get("job_ids") or [])
            for r in successful
        )

        def _stats(values):
            vs = [v for v in values if v is not None]
            if not vs:
                return None
            arr = np.asarray(vs, dtype=np.float64)
            return {
                "n": len(vs),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
            }

        valid_rates = [
            np.mean([t.get("valid_rate") for t in (r.get("per_tau") or [])])
            if r.get("per_tau") else None
            for r in successful
        ]
        accepted_rates = [
            np.mean([t.get("acceptance_rate") for t in (r.get("per_tau") or [])])
            if r.get("per_tau") else None
            for r in successful
        ]
        top1_rmsds = [
            (r.get("structure", {}).get("top1") or {}).get("rmsd_to_reference")
            for r in successful
        ]
        best_rmsds = [
            (r.get("landscape", {}).get("funnel") or {}).get("best_rmsd")
            for r in successful
        ]
        n_basins = [
            r.get("post", {}).get("n_basins") for r in successful
        ]

        backend_str = None
        if successful:
            backend_str = successful[0].get("backend", {}).get("backend_type")

        return {
            "backend": backend_str,
            "n_tasks": n_total,
            "n_success": len(successful),
            "n_failed": len(failed),
            "total_shots": int(total_shots),
            "total_ibm_jobs": int(total_jobs),
            "valid_rate_distribution": _stats(valid_rates),
            "accepted_rate_distribution": _stats(accepted_rates),
            "top1_rmsd_distribution": _stats(top1_rmsds),
            "best_rmsd_distribution": _stats(best_rmsds),
            "basin_count_distribution": _stats(n_basins),
            "failed_tasks": failed,
        }


# ---------------------------------------------------------------------- #
# misc helpers                                                           #
# ---------------------------------------------------------------------- #

def _csv_val(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def _read_backend_status(
    quantum_dir: Path,
) -> Tuple[Optional[str], Optional[List[str]]]:
    p = quantum_dir / "backend_result.json"
    if not p.exists():
        return None, None
    try:
        d = json.loads(p.read_text())
        return d.get("status"), d.get("job_ids")
    except Exception:
        return None, None


def _write_per_tau_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "tau", "n_raw", "n_valid", "n_invalid", "n_accepted",
        "valid_rate", "acceptance_rate", "fallback_triggered",
        "long_range_clash",
        "mean_filter_energy_accepted", "mean_full_energy_accepted",
        "favorable_contact_miss_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _csv_val(r.get(k)) for k in cols})


def _write_candidate_csv(
    path: Path,
    samples: List[CandidateSample],
    *,
    include_full_energy: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "bitstring", "count", "base_probability",
        "filter_energy", "full_energy", "valid", "accepted",
        "is_dense",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for s in samples:
            w.writerow({
                "bitstring": s.bitstring or "",
                "count": s.count,
                "base_probability": _csv_val(s.base_probability),
                "filter_energy": _csv_val(s.filter_energy),
                "full_energy": _csv_val(s.full_energy),
                "valid": "true" if s.valid else "false",
                "accepted": "true" if s.accepted else "false",
                "is_dense": "true" if _is_dense(s) else "false",
            })


def _write_refined_csv(
    path: Path, refined_candidates,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "rank", "bitstring", "refined_score", "refined_weight",
        "full_energy", "filter_energy", "is_dense", "basin_id",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for rk, c in enumerate(refined_candidates):
            s = c.sample
            w.writerow({
                "rank": rk,
                "bitstring": s.bitstring or "",
                "refined_score": f"{c.refined_score:.6f}",
                "refined_weight": f"{c.refined_weight:.6f}",
                "full_energy": _csv_val(s.full_energy),
                "filter_energy": _csv_val(s.filter_energy),
                "is_dense": "true" if _is_dense(s) else "false",
                "basin_id": "" if c.basin_id is None else int(c.basin_id),
            })


def _write_candidates_coords_archive(
    *,
    task_dir: Path,
    accepted_pool: List[CandidateSample],
    dense_candidates: List[CandidateSample],
    refined_candidates,
) -> None:
    """Save coordinates for every candidate (accepted + dense + refined)
    so the post-hoc oracle eval can compute oracle-best RMSD without
    re-running sampling. Sources may produce duplicate coords arrays;
    that's OK — the index records the source.

    Layout:
        candidates/all_candidates_index.csv
        candidates/all_candidates_coords.npz  (key candidate_uid → ndarray)
    """
    out_dir = task_dir / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)

    coords_map: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []

    def _row(uid, source, sample, *,
             refined_score=None, refined_weight=None, basin_id=None):
        d = sample.metadata.get("densify") or {}
        return {
            "candidate_uid": uid,
            "source": source,
            "bitstring": sample.bitstring or "",
            "is_dense": "true" if d.get("is_perturbed") else "false",
            "parent_bitstring": d.get("parent_bitstring") or "",
            "full_energy": (
                "" if sample.full_energy is None
                else f"{sample.full_energy:.6f}"
            ),
            "filter_energy": (
                "" if sample.filter_energy is None
                else f"{sample.filter_energy:.6f}"
            ),
            "refined_score": (
                "" if refined_score is None
                else f"{float(refined_score):.6f}"
            ),
            "refined_weight": (
                "" if refined_weight is None
                else f"{float(refined_weight):.6f}"
            ),
            "count": int(sample.count),
            "tau": (
                "" if sample.metadata.get("tau") is None
                else f"{float(sample.metadata['tau']):.4f}"
            ),
            "basin_id": "" if basin_id is None else str(int(basin_id)),
            "coords_key": uid,
        }

    for i, s in enumerate(accepted_pool):
        if s.coords is None:
            continue
        uid = f"acc_{i:05d}"
        coords_map[uid] = np.asarray(s.coords, dtype=np.float64)
        rows.append(_row(uid, "accepted", s))

    for i, s in enumerate(dense_candidates):
        if s.coords is None:
            continue
        uid = f"dense_{i:05d}"
        coords_map[uid] = np.asarray(s.coords, dtype=np.float64)
        rows.append(_row(uid, "dense", s))

    for i, c in enumerate(refined_candidates):
        s = c.sample
        if s.coords is None:
            continue
        uid = f"refined_{i:05d}"
        coords_map[uid] = np.asarray(s.coords, dtype=np.float64)
        rows.append(_row(
            uid, "refined", s,
            refined_score=c.refined_score,
            refined_weight=c.refined_weight,
            basin_id=c.basin_id,
        ))

    cols = [
        "candidate_uid", "source", "bitstring", "is_dense",
        "parent_bitstring", "full_energy", "filter_energy",
        "refined_score", "refined_weight", "count", "tau",
        "basin_id", "coords_key",
    ]
    idx_path = out_dir / "all_candidates_index.csv"
    with idx_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if coords_map:
        np.savez_compressed(
            out_dir / "all_candidates_coords.npz", **coords_map,
        )


__all__ = ["KrasFullBatchRunner"]
