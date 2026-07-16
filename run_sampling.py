# Author: Yuqi Zhang
"""Stage 1 — pipeline sampling entry (stages A–G).

For the COMPLETE flow (sampling + reconstruction + oracle + docking +
validation) on any system, use ``run_pipeline.py``, which orchestrates
this script. Run this directly only when you want the sampling stages
alone.

This drives the seven-stage pipeline (``doc/SYSTEM_REPORT.md``):

    A. env-conditioned V2 moment matching  (build_environment_prior /
       build_corridor_prior wired into MomentMatchInitializer)
    B. closed-form HEA θ                    (no VQE)
    C. HEA circuit assembly                 (ansatz="hea_with_tf")
    D. quantum sampling                     (Aer or IBM Runtime)
    E. classical imaginary-time rejection   (exp(-τ·H_filter))
    F. Hybrid SQD refinement                (coupling_mode="hybrid")
    G. postprocess + landscape

Running

    python run_sampling.py

with NO further arguments runs the pipeline on the LOCAL Aer
simulator (safe default) over ``inputs/kras_tasks.csv``. It is
equivalent to:

    python run_sampling.py \\
        --tasks inputs/kras_tasks.csv \\
        --structure-root kras_select_systems \\
        --backend aer \\
        --n-circuits 4 \\
        --shots 2048 \\
        --taus 0.0,0.1,0.2 \\
        --run-name pipeline_v1

To submit to real IBM hardware you must OPT IN explicitly (nothing is
submitted by accident):

    python run_sampling.py \\
        --backend ibm --submit-ibm \\
        --backend-name ibm_cleveland \\
        --execution-mode batch \\
        --n-circuits 8 --shots 2048 \\
        --run-name pipeline_ibm_v1

``--backend ibm`` without ``--submit-ibm`` stays in IBM Runtime dry-run
mode (transpile / cost estimate only; no job submission).

Scope: this script is sampling + postprocess + landscape only. It does
NOT run PULCHRA reconstruction or Vina docking — those are handled by
``run_oracle_docking_eval.py`` and therefore this entry does not require
``external_tools.json`` / PULCHRA / OpenBabel / Vina.

This script is a thin CLI wrapper around
``ras_folding.kras.full_batch_runner.KrasFullBatchRunner`` — it does NOT
duplicate the per-task pipeline logic.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ras_folding.kras.full_batch_runner import (
    KrasFullBatchRunner,
    _RunnerDefaults,
)
from ras_folding.kras.task_loader import load_kras_tasks
from ras_folding.quantum.backend_config import QuantumBackendConfig


# Anchor files / dirs that MUST exist in the project root for a
# full-pipeline run to make sense. These are the same paths the user
# will reference from a PyCharm Run Configuration.
_REQUIRED_PROJECT_ANCHORS = (
    "inputs/kras_tasks.csv",
    "kras_select_systems",
    "ras_folding",
    "run_sampling.py",
)


def _parse_taus(s) -> Tuple[float, ...]:
    if isinstance(s, (list, tuple)):
        return tuple(float(t) for t in s)
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--taus must contain at least one value")
    return tuple(float(p) for p in parts)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "KRAS pipeline entry. Defaults to the LOCAL Aer "
            "simulator. Pass --backend ibm --submit-ibm to opt in to a "
            "real IBM Runtime submission."
        )
    )
    p.add_argument("--tasks", type=str, default="inputs/kras_tasks.csv")
    p.add_argument("--structure-root", type=str, default="kras_select_systems")
    p.add_argument("--output-root", type=str, default="logs/kras_full_batch")
    p.add_argument("--run-name", type=str, default="pipeline_v1")

    p.add_argument(
        "--backend", type=str, default="aer",
        choices=("aer", "ibm"),
        help="Quantum backend. Default 'aer' (local, safe).",
    )
    p.add_argument(
        "--submit-ibm",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Submit to IBM Runtime when --backend ibm. Default False "
             "(dry-run). Pass --submit-ibm to opt in to a real submission.",
    )
    p.add_argument("--backend-name", type=str, default="ibm_cleveland")
    p.add_argument(
        "--execution-mode", type=str, default="batch",
        choices=("job", "batch"),
    )
    p.add_argument("--n-circuits", type=int, default=4)
    p.add_argument("--shots", type=int, default=2048)
    p.add_argument("--taus", type=_parse_taus, default="0.0,0.1,0.2")
    p.add_argument("--max-tasks", type=int, default=None)

    p.add_argument(
        "--overwrite", action="store_true",
        help=(
            "Default False = resume mode (skip task_dirs that already "
            "have a DONE marker). Pass --overwrite to force re-run."
        ),
    )
    p.add_argument("--no-densify", action="store_true")
    p.add_argument("--no-landscape", action="store_true")
    p.add_argument("--no-postprocess", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument(
        "--paired-tau-sampling",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return p


# ---------------------------------------------------------------------- #
# preflight                                                              #
# ---------------------------------------------------------------------- #

def check_project_root(project_root: Path) -> Dict[str, Any]:
    """Verify the project root contains the anchor paths a full run
    needs. Returns a dict; never raises. The caller decides whether to
    abort on ``ok=False``."""
    project_root = Path(project_root).resolve()
    missing: List[str] = []
    for rel in _REQUIRED_PROJECT_ANCHORS:
        if not (project_root / rel).exists():
            missing.append(rel)
    cwd = Path.cwd().resolve()
    return {
        "project_root": str(project_root),
        "cwd": str(cwd),
        "cwd_matches_project_root": cwd == project_root,
        "missing_anchors": missing,
        "ok": not missing,
    }


def _qubit_distribution(tasks: List[Any]) -> Dict[str, Any]:
    """Bucket tasks by n_residues / n_qubits so the user can sanity-
    check what they're about to submit. ``n_qubits = n_bonds * 6`` per
    the encoder convention."""
    n_residues_hist: Counter = Counter()
    n_qubits_hist: Counter = Counter()
    n_with_ref = 0
    for t in tasks:
        nr = int(getattr(t.encoder_inputs, "n_residues", 0))
        nb = int(getattr(t.encoder_inputs, "n_bonds", max(0, nr - 1)))
        n_residues_hist[nr] += 1
        n_qubits_hist[nb * 6] += 1
        if getattr(t, "reference_coords", None) is not None:
            n_with_ref += 1
    return {
        "n_tasks": len(tasks),
        "n_with_reference": n_with_ref,
        "n_residues": dict(sorted(n_residues_hist.items())),
        "n_qubits": dict(sorted(n_qubits_hist.items())),
    }


# ---------------------------------------------------------------------- #
# build runner                                                           #
# ---------------------------------------------------------------------- #

def build_runner(args: argparse.Namespace, project_root: Path):
    """Resolve all CLI arguments into (runner, tasks, schema, plan).

    Factored out so tests can call this without invoking runner.run().
    """
    if args.backend == "aer":
        # AER never submits; force submit_ibm=False regardless of CLI.
        submit_ibm = False
    else:
        submit_ibm = bool(args.submit_ibm)

    if args.backend == "aer":
        if args.paired_tau_sampling is None:
            paired = True
        else:
            paired = bool(args.paired_tau_sampling)
        backend_config = QuantumBackendConfig(
            backend_type="aer_simulator",
            shots_per_circuit=int(args.shots),
            seed_simulator=int(args.seed),
            seed_transpiler=int(args.seed),
            paired_tau_sampling=paired,
            dry_run=False,
        )
    else:
        if args.paired_tau_sampling is None:
            paired = (not submit_ibm)
        else:
            paired = bool(args.paired_tau_sampling)
        backend_config = QuantumBackendConfig(
            backend_type="ibm_runtime",
            ibm_backend_name=args.backend_name,
            execution_mode=args.execution_mode,
            shots_per_circuit=int(args.shots),
            seed_transpiler=int(args.seed),
            paired_tau_sampling=paired,
            dry_run=(not submit_ibm),
        )

    csv_path = (
        Path(args.tasks) if Path(args.tasks).is_absolute()
        else project_root / args.tasks
    )
    pdb_dir = (
        Path(args.structure_root) if Path(args.structure_root).is_absolute()
        else project_root / args.structure_root
    )
    tasks, schema = load_kras_tasks(
        csv_path, pdb_dir=pdb_dir, flank_size=1, skip_missing_pdb=True,
    )

    defaults = _RunnerDefaults(
        taus=tuple(args.taus),
        n_circuits=int(args.n_circuits),
        seed=int(args.seed),
    )
    output_root = (
        Path(args.output_root) if Path(args.output_root).is_absolute()
        else project_root / args.output_root
    )
    runner = KrasFullBatchRunner(
        backend_config=backend_config,
        # the runner forces ansatz="hea_with_tf" and injects the
        # moment-matched θ itself; we only seed the builder here.
        circuit_builder_config={"seed": int(args.seed)},
        # pdb_dir enables Stage A: _build_v2_prior reads the reference PDB
        # to build the env + corridor prior fed into moment matching.
        pdb_dir=pdb_dir,
        output_root=output_root,
        run_name=args.run_name,
        overwrite=bool(args.overwrite),
        fail_fast=bool(args.fail_fast),
        densify_enabled=not args.no_densify,
        postprocess_enabled=not args.no_postprocess,
        landscape_enabled=not args.no_landscape,
        max_tasks=args.max_tasks,
        defaults=defaults,
    )

    n_runnable = (
        len(tasks) if args.max_tasks is None
        else min(len(tasks), int(args.max_tasks))
    )
    estimated_circuits = (
        n_runnable * int(args.n_circuits) * len(args.taus)
    )
    estimated_total_shots = estimated_circuits * int(args.shots)
    plan = {
        "submit_ibm": submit_ibm,
        "backend": args.backend,
        "backend_name": args.backend_name,
        "execution_mode": args.execution_mode,
        "paired_tau_sampling": paired,
        "n_tasks": n_runnable,
        "n_circuits": int(args.n_circuits),
        "shots_per_circuit": int(args.shots),
        "taus": tuple(args.taus),
        "estimated_circuits": estimated_circuits,
        "estimated_total_shots": estimated_total_shots,
        "estimated_job_groups": (
            n_runnable if args.execution_mode == "batch" else estimated_circuits
        ),
        "run_name": args.run_name,
        "output_root": str(output_root),
        "run_dir": str(output_root / args.run_name),
        "overwrite": bool(args.overwrite),
        "resume_enabled": (not bool(args.overwrite)),
        "densify_enabled": not args.no_densify,
        "postprocess_enabled": not args.no_postprocess,
        "landscape_enabled": not args.no_landscape,
        # sampling-only scope: sampling+postprocess+landscape only.
        "reconstruction_included": False,
        "docking_included": False,
    }
    return runner, tasks, schema, plan


# ---------------------------------------------------------------------- #
# pretty-printers                                                        #
# ---------------------------------------------------------------------- #

def print_preflight(
    root_check: Dict[str, Any],
    schema: Dict[str, Any],
    qdist: Dict[str, Any],
    plan: Dict[str, Any],
) -> None:
    print("=" * 78)
    print("[PREFLIGHT] project root + inputs")
    print(f"  project_root            = {root_check['project_root']}")
    print(f"  cwd                     = {root_check['cwd']}")
    print(f"  cwd_matches_root        = {root_check['cwd_matches_project_root']}")
    if root_check["missing_anchors"]:
        print(f"  missing_anchors         = {root_check['missing_anchors']}")
    n_skipped = len(schema.get("rows_skipped_in_loader") or [])
    n_csv_total = int(schema.get("n_loaded", 0)) + n_skipped
    print(f"  n_tasks_total           = {n_csv_total}")
    print(f"  n_tasks_loadable        = {schema.get('n_loaded')}")
    print(f"  n_tasks_with_reference  = {schema.get('n_with_reference')}")
    print(f"  n_tasks_missing_struct  = {n_skipped}")
    print(f"  n_residues_distribution = {qdist['n_residues']}")
    print(f"  n_qubits_distribution   = {qdist['n_qubits']}")
    print()
    print("[PREFLIGHT] backend / sampling ")
    print(f"  sampler                 = hea_moment_matched (HEA + V2 θ)")
    print(f"  refiner_coupling        = hybrid (Pauli ⊕ RMSD SQD)")
    print(f"  backend                 = {plan['backend']}")
    print(f"  backend_name            = {plan['backend_name']}")
    print(f"  execution_mode          = {plan['execution_mode']}")
    print(f"  submit_ibm              = {plan['submit_ibm']}")
    print(f"  paired_tau_sampling     = {plan['paired_tau_sampling']}")
    print(f"  n_circuits              = {plan['n_circuits']}")
    print(f"  shots_per_circuit       = {plan['shots_per_circuit']}")
    print(f"  taus                    = {plan['taus']}")
    print(f"  estimated_circuits      = {plan['estimated_circuits']}")
    print(f"  estimated_total_shots   = {plan['estimated_total_shots']}")
    print(f"  estimated_job_groups    = {plan['estimated_job_groups']}")
    print()
    print("[PREFLIGHT] outputs / resume")
    print(f"  run_name                = {plan['run_name']}")
    print(f"  run_dir                 = {plan['run_dir']}")
    print(f"  overwrite               = {plan['overwrite']}")
    print(f"  resume_enabled          = {plan['resume_enabled']}")
    print(f"  densify_enabled         = {plan['densify_enabled']}")
    print(f"  postprocess_enabled     = {plan['postprocess_enabled']}")
    print(f"  landscape_enabled       = {plan['landscape_enabled']}")
    print(f"  reconstruction_included = {plan['reconstruction_included']}")
    print(f"  docking_included        = {plan['docking_included']}")
    run_dir = Path(plan["run_dir"])
    if run_dir.exists():
        if plan["overwrite"]:
            print(
                f"  NOTE: run_dir EXISTS and overwrite=True — "
                "existing tasks will be RE-RUN."
            )
        else:
            print(
                f"  NOTE: run_dir EXISTS and overwrite=False — "
                "tasks with a DONE marker will be SKIPPED (resume mode)."
            )
    else:
        print(f"  NOTE: run_dir does not exist yet (fresh run).")
    print("=" * 78)
    if plan["submit_ibm"]:
        print()
        print("################################################################")
        print("##  REAL FULL IBM RUN WILL BE SUBMITTED                       ##")
        print(f"##  backend               = {plan['backend_name']}")
        print(f"##  execution_mode        = {plan['execution_mode']}")
        print(f"##  n_tasks               = {plan['n_tasks']}")
        print(f"##  n_circuits            = {plan['n_circuits']}")
        print(f"##  shots                 = {plan['shots_per_circuit']}")
        print(f"##  taus                  = {plan['taus']}")
        print(f"##  estimated_total_shots = {plan['estimated_total_shots']}")
        print(f"##  run_name              = {plan['run_name']}")
        print(f"##  output_dir            = {plan['run_dir']}")
        print("################################################################")


# Backwards-compatible wrapper kept for tests / external callers that
# only want the plan block.
def print_run_plan(plan: Dict[str, Any]) -> None:
    print("=== run_sampling plan ===")
    print(f"  backend             = {plan['backend']}")
    print(f"  submit_ibm          = {plan['submit_ibm']}")
    print(f"  backend_name        = {plan['backend_name']}")
    print(f"  execution_mode      = {plan['execution_mode']}")
    print(f"  n_tasks             = {plan['n_tasks']}")
    print(f"  n_circuits          = {plan['n_circuits']}")
    print(f"  shots_per_circuit   = {plan['shots_per_circuit']}")
    print(f"  taus                = {plan['taus']}")
    print(f"  estimated_circuits  = {plan['estimated_circuits']}")
    print(f"  estimated_total_shots= {plan['estimated_total_shots']}")
    print(f"  run_name            = {plan['run_name']}")
    print(f"  output_root         = {plan['output_root']}")


# ---------------------------------------------------------------------- #
# per-task progress callback                                             #
# ---------------------------------------------------------------------- #

def make_progress_callback(
    tasks: List[Any],
    plan: Dict[str, Any],
) -> Callable[[str, Dict[str, Any]], None]:
    """Build a runner ``progress_callback`` that prints [RUN] / [DONE]
    / [SKIP] / [ERROR] lines per the agreed format."""
    n_total = len(tasks)
    by_id = {t.task_id: t for t in tasks}
    start_times: Dict[str, float] = {}

    def _cb(event: str, payload: Dict[str, Any]) -> None:
        i = int(payload.get("i", 0))
        tid = str(payload.get("task_id", ""))
        task = by_id.get(tid)
        if event == "task_start":
            start_times[tid] = time.time()
            seq_len = (
                int(getattr(task.encoder_inputs, "n_residues", 0))
                if task is not None else 0
            )
            n_bonds = (
                int(getattr(task.encoder_inputs, "n_bonds", max(0, seq_len - 1)))
                if task is not None else 0
            )
            print()
            print(
                f"[RUN] {i+1:02d}/{n_total} {tid}"
            )
            print(f"  sequence_length    : {seq_len}")
            print(f"  n_qubits           : {n_bonds * 6}")
            print(f"  backend            : {plan['backend_name']}")
            print(f"  taus               : {plan['taus']}")
            print(f"  n_circuits         : {plan['n_circuits']}")
            print(f"  shots/circuit      : {plan['shots_per_circuit']}")
            print(f"  output             : {payload.get('task_dir')}")
            sys.stdout.flush()
        elif event == "task_done":
            elapsed = float(payload.get("elapsed_sec", 0.0))
            print(f"[DONE] {tid}  (elapsed {elapsed:.1f}s)")
            print(f"  output_dir         : {payload.get('task_dir')}")
            sys.stdout.flush()
        elif event == "task_skipped":
            print(
                f"[SKIP] {i+1:02d}/{n_total} {tid}  (DONE marker present; "
                "pass --overwrite to re-run)"
            )
            sys.stdout.flush()
        elif event == "task_failed":
            print(f"[ERROR] {tid}")
            print(f"  error              : {payload.get('error')}")
            print(f"  error_path         : {payload.get('task_dir')}/ERROR.txt")
            print(f"  continuing         : True")
            sys.stdout.flush()
    return _cb


def print_global_summary(
    plan: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> None:
    n_success = sum(
        1 for r in results
        if not r.get("__failed") and not r.get("__skipped")
    )
    n_skipped = sum(1 for r in results if r.get("__skipped"))
    n_failed = sum(1 for r in results if r.get("__failed"))
    print()
    print("=" * 78)
    print("[GLOBAL DONE]")
    print(f"  n_success              = {n_success}")
    print(f"  n_skipped (resume)     = {n_skipped}")
    print(f"  n_failed               = {n_failed}")
    run_dir = Path(plan["run_dir"])
    print(f"  global_summary.csv     = {run_dir / 'global_summary.csv'}")
    print(f"  global_report.md       = {run_dir / 'global_report.md'}")
    print(f"  failed_tasks.csv       = {run_dir / 'failed_tasks.csv'}")
    print()
    print("Next steps:")
    print()
    print("1. Run oracle + docking evaluation (PULCHRA + Vina required):")
    print(
        "   python check_external_tools.py --config external_tools.json"
    )
    print(
        "   python run_oracle_docking_eval.py \\"
    )
    print(f"     --run-dir {run_dir} \\")
    print(f"     --tasks inputs/kras_tasks.csv \\")
    print(f"     --structure-root kras_select_systems \\")
    print(f"     --external-tools-config external_tools.json")
    print("=" * 78)


# ---------------------------------------------------------------------- #
# main                                                                   #
# ---------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    project_root = Path(_HERE)

    # Project-root anchor check FIRST. If the user runs the script from
    # a wrong cwd in PyCharm, the loader would otherwise produce a less
    # actionable error later. We use _HERE (= the script's directory)
    # as project_root, and only WARN when cwd != project_root rather
    # than erroring (PyCharm sometimes runs with cwd = parent dir).
    root_check = check_project_root(project_root)
    if not root_check["ok"]:
        print("=" * 78)
        print("[PREFLIGHT] FAILED — project root missing required anchors:")
        for rel in root_check["missing_anchors"]:
            print(f"   - {rel}")
        print(f"   project_root probed: {root_check['project_root']}")
        print(f"   cwd               : {root_check['cwd']}")
        print(
            "   Fix: set the PyCharm Run Configuration's "
            "'Working directory' to the project root."
        )
        print("=" * 78)
        raise SystemExit(2)
    if not root_check["cwd_matches_project_root"]:
        print(
            "[PREFLIGHT] WARNING: cwd != project_root. Relative paths "
            f"resolve against {root_check['project_root']!r}, not "
            f"{root_check['cwd']!r}."
        )

    runner, tasks, schema, plan = build_runner(args, project_root)
    qdist = _qubit_distribution(tasks)
    print_preflight(root_check, schema, qdist, plan)

    # Wire per-task progress printing
    runner.progress_callback = make_progress_callback(tasks, plan)

    results = runner.run(tasks, schema_summary=schema)

    print_global_summary(plan, results or [])


if __name__ == "__main__":
    main()
