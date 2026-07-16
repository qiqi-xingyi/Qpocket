# Author: Yuqi Zhang
"""run_pipeline.py — unified, system-agnostic FULL-FLOW entry.

ONE command runs the COMPLETE pipeline on ANY protein system:

    sampling (stages A–G)
        -> reconstruction (PULCHRA all-atom)
        -> oracle (RMSD-best candidate vs native)
        -> docking (AutoDock Vina, auto-skipped if vina absent)
        -> validation (PASS / WARN / FAIL diagnostic)

It is NOT tied to KRAS — point it at any PDB + fragment definition.

Two ways to specify the system
------------------------------

(1) Single system, no CSV needed — just a structure + a residue range:

      python run_pipeline.py \
          --pdb /path/to/structure.pdb --chain A \
          --start-resi 10 --end-resi 20 [--ligand LIG] \
          --run-name my_run

(2) Batch from a tasks CSV (any structures). The CSV needs at least the
    columns ``ref_pdb, chain_id, start_resi, end_resi`` (``ligand_resname``
    optional; sequence is read from the PDB). ``--structure-root`` is the
    directory holding the referenced PDBs:

      python run_pipeline.py \
          --tasks my_tasks.csv --structure-root /path/to/pdbs \
          --run-name my_batch

Quantum backend
---------------
Defaults to the LOCAL Aer simulator. Real IBM hardware is opt-in:

      python run_pipeline.py --pdb ... --backend ibm --submit-ibm \
          --backend-name ibm_cleveland

Flow control
------------
- ``--skip-eval``            : sampling only (stages A–G).
- ``--skip-docking``         : reconstruction + oracle, no Vina docking
                               (auto-enabled when vina/obabel are missing).
- ``--skip-reconstruction``  : oracle only (implies skip-docking).
- ``--no-validation``        : skip the final PASS/WARN/FAIL pass.
- ``--skip-sampling``        : reuse an existing sampling run_dir and only
                               run the downstream eval + validation.

This script is a thin orchestrator: it reuses
``run_sampling.build_runner`` (sampling),
``run_oracle_docking_eval.run_evaluation`` (downstream), the shared
``run_external_tools_preflight`` (tool detection) and
``pipeline_validation.FinalPipelineValidator`` (validation). It adds no
new pipeline logic of its own.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import run_sampling as rpr
import run_oracle_docking_eval as rode
from ras_folding.external_tools import run_external_tools_preflight
from pipeline_validation.checks import FinalPipelineValidator
from pipeline_validation.report import (
    write_validation_summary, write_validation_report,
)


# ---------------------------------------------------------------------- #
# CLI                                                                    #
# ---------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Unified full-flow entry: sampling -> reconstruction -> oracle "
            "-> docking -> validation, on ANY protein system. Local Aer by "
            "default; IBM is opt-in."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sysg = p.add_argument_group("system input (choose ONE of A or B)")
    # (A) single-system
    sysg.add_argument("--pdb", type=str, default=None,
                      help="(A) single system: path to a PDB structure.")
    sysg.add_argument("--chain", type=str, default=None,
                      help="(A) chain id of the fragment.")
    sysg.add_argument("--start-resi", type=int, default=None,
                      help="(A) first residue number of the fragment.")
    sysg.add_argument("--end-resi", type=int, default=None,
                      help="(A) last residue number of the fragment.")
    sysg.add_argument("--ligand", type=str, default=None,
                      help="(A) optional ligand resname for the pocket prior "
                           "and docking box (e.g. EZZ).")
    sysg.add_argument("--task-id", type=str, default=None,
                      help="(A) optional task id; default derived from "
                           "pdb/chain/range.")
    sysg.add_argument("--scale-factor", type=float, default=3.8,
                      help="(A) CA-CA virtual bond length (Angstrom).")
    sysg.add_argument("--scale-mode", type=str, default="fixed",
                      help="(A) scale mode for the encoder.")
    # (B) batch CSV
    sysg.add_argument("--tasks", type=str, default=None,
                      help="(B) batch: tasks CSV (ref_pdb,chain_id,start_resi,"
                           "end_resi[,ligand_resname]).")
    sysg.add_argument("--structure-root", type=str, default=None,
                      help="(B) directory holding the PDBs referenced by the "
                           "tasks CSV.")

    qg = p.add_argument_group("quantum backend / sampling")
    qg.add_argument("--backend", type=str, default="aer",
                    choices=("aer", "ibm"))
    qg.add_argument("--submit-ibm", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Opt in to a REAL IBM Runtime submission "
                         "(only with --backend ibm).")
    qg.add_argument("--backend-name", type=str, default="ibm_cleveland")
    qg.add_argument("--execution-mode", type=str, default="batch",
                    choices=("job", "batch"))
    qg.add_argument("--n-circuits", type=int, default=4)
    qg.add_argument("--shots", type=int, default=2048)
    qg.add_argument("--taus", type=rpr._parse_taus, default="0.0,0.1,0.2")
    qg.add_argument("--seed", type=int, default=2024)
    qg.add_argument("--paired-tau-sampling",
                    action=argparse.BooleanOptionalAction, default=None)
    qg.add_argument("--no-densify", action="store_true")
    qg.add_argument("--no-landscape", action="store_true")
    qg.add_argument("--no-postprocess", action="store_true")

    dg = p.add_argument_group("downstream eval (reconstruction + docking)")
    dg.add_argument("--external-tools-config", type=str,
                    default="external_tools.json")
    dg.add_argument("--pulchra-bin", type=str, default=None)
    dg.add_argument("--obabel-bin", type=str, default=None)
    dg.add_argument("--vina-bin", type=str, default=None)
    dg.add_argument("--repeats", type=int, default=5)
    dg.add_argument("--exhaustiveness", type=int, default=8)
    dg.add_argument("--num-modes", type=int, default=9)
    dg.add_argument("--temperature-k", type=float, default=298.15)
    dg.add_argument(
        "--no-kabsch-rmsd",
        dest="use_kabsch_rmsd",
        action="store_false",
        help=(
            "Disable Kabsch alignment when computing oracle RMSD. Off by "
            "default: candidate coords come from anchor decoding, so "
            "unaligned RMSD is meaningless."
        ),
    )
    dg.set_defaults(use_kabsch_rmsd=True)

    fg = p.add_argument_group("flow control")
    fg.add_argument("--skip-sampling", action="store_true",
                    help="Reuse an existing sampling run_dir; run only the "
                         "downstream eval + validation.")
    fg.add_argument("--skip-eval", action="store_true",
                    help="Sampling only (stages A-G); no downstream eval.")
    fg.add_argument("--skip-docking", action="store_true")
    fg.add_argument("--skip-reconstruction", action="store_true")
    fg.add_argument("--no-validation", action="store_true")

    gg = p.add_argument_group("general")
    gg.add_argument("--output-root", type=str, default="logs/pipeline")
    gg.add_argument("--run-name", type=str, default="pipeline_v1")
    gg.add_argument("--max-tasks", type=int, default=None)
    gg.add_argument("--overwrite", action="store_true")
    gg.add_argument("--fail-fast", action="store_true")
    return p


# ---------------------------------------------------------------------- #
# system resolution                                                      #
# ---------------------------------------------------------------------- #

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


def resolve_system(
    args: argparse.Namespace, project_root: Path, run_dir: Path,
) -> Tuple[Path, Path]:
    """Resolve the system spec into (tasks_csv, structure_root).

    Single-system mode (``--pdb``) synthesizes a one-row tasks CSV under
    ``run_dir`` and uses the PDB's parent directory as the structure root.
    Batch mode uses ``--tasks`` / ``--structure-root`` directly.
    Returns absolute paths.
    """
    if args.pdb:
        missing = [
            n for n, v in (("--chain", args.chain),
                            ("--start-resi", args.start_resi),
                            ("--end-resi", args.end_resi))
            if v is None
        ]
        if missing:
            raise SystemExit(
                f"single-system mode (--pdb) also requires {missing}"
            )
        pdb_path = Path(args.pdb)
        if not pdb_path.is_absolute():
            pdb_path = (project_root / pdb_path)
        pdb_path = pdb_path.resolve()
        if not pdb_path.is_file():
            raise SystemExit(f"--pdb not found: {pdb_path}")
        structure_root = pdb_path.parent
        task_id = args.task_id or _slug(
            f"{pdb_path.stem}_{args.chain}_{args.start_resi}_{args.end_resi}"
        )
        csv_path = run_dir / "_system_task.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow([
                "case_id", "ref_pdb", "chain_id", "start_resi", "end_resi",
                "ligand_resname", "scale_factor", "scale_mode",
                "has_native_ref",
            ])
            w.writerow([
                task_id, pdb_path.name, args.chain,
                int(args.start_resi), int(args.end_resi),
                args.ligand or "", float(args.scale_factor),
                args.scale_mode, "true",
            ])
        print(f"[system] single-system mode")
        print(f"[system]   pdb            = {pdb_path}")
        print(f"[system]   fragment       = chain {args.chain} "
              f"{args.start_resi}-{args.end_resi}"
              + (f", ligand {args.ligand}" if args.ligand else ""))
        print(f"[system]   task_id        = {task_id}")
        print(f"[system]   synthesized    = {csv_path}")
        return csv_path, structure_root

    # batch CSV mode
    if not args.tasks:
        raise SystemExit(
            "provide a system: either --pdb (+ --chain/--start-resi/"
            "--end-resi) for a single system, or --tasks CSV (+ "
            "--structure-root) for a batch."
        )
    csv_path = Path(args.tasks)
    if not csv_path.is_absolute():
        csv_path = (project_root / csv_path)
    csv_path = csv_path.resolve()
    if not csv_path.is_file():
        raise SystemExit(f"--tasks CSV not found: {csv_path}")
    if args.structure_root:
        structure_root = Path(args.structure_root)
        if not structure_root.is_absolute():
            structure_root = (project_root / structure_root)
        structure_root = structure_root.resolve()
    else:
        # default: a sibling 'kras_select_systems' or the CSV's dir
        cand = project_root / "kras_select_systems"
        structure_root = cand if cand.is_dir() else csv_path.parent
    if not structure_root.is_dir():
        raise SystemExit(f"--structure-root not found: {structure_root}")
    print(f"[system] batch mode")
    print(f"[system]   tasks_csv      = {csv_path}")
    print(f"[system]   structure_root = {structure_root}")
    return csv_path, structure_root


# ---------------------------------------------------------------------- #
# stage 1: sampling (A-G), reusing run_sampling            #
# ---------------------------------------------------------------------- #

def run_sampling(
    args: argparse.Namespace, project_root: Path,
    tasks_csv: Path, structure_root: Path,
) -> Tuple[Path, List[Any]]:
    """Run stages A-G via run_sampling.build_runner. Returns
    (run_dir, tasks)."""
    ns = Namespace(
        tasks=str(tasks_csv), structure_root=str(structure_root),
        output_root=args.output_root, run_name=args.run_name,
        backend=args.backend, submit_ibm=args.submit_ibm,
        backend_name=args.backend_name, execution_mode=args.execution_mode,
        n_circuits=args.n_circuits, shots=args.shots, taus=args.taus,
        max_tasks=args.max_tasks, overwrite=args.overwrite,
        no_densify=args.no_densify, no_landscape=args.no_landscape,
        no_postprocess=args.no_postprocess, fail_fast=args.fail_fast,
        seed=args.seed, paired_tau_sampling=args.paired_tau_sampling,
    )
    runner, tasks, schema, plan = rpr.build_runner(ns, project_root)
    qdist = rpr._qubit_distribution(tasks)
    rpr.print_preflight(rpr.check_project_root(project_root), schema, qdist, plan)
    runner.progress_callback = rpr.make_progress_callback(tasks, plan)
    results = runner.run(tasks, schema_summary=schema) or []
    # NB: we deliberately do NOT call rpr.print_global_summary here — its
    # "next steps" block tells the user to run the downstream eval manually,
    # which this unified entry does automatically in stage 3.
    n_ok = sum(1 for r in results if not r.get("__failed") and not r.get("__skipped"))
    n_skip = sum(1 for r in results if r.get("__skipped"))
    n_fail = sum(1 for r in results if r.get("__failed"))
    print(f"  sampling done: success={n_ok} skipped={n_skip} failed={n_fail}")
    return Path(plan["run_dir"]), tasks


# ---------------------------------------------------------------------- #
# stage 2: external-tool preflight + skip decisions                     #
# ---------------------------------------------------------------------- #

def resolve_downstream_tools(
    args: argparse.Namespace, project_root: Path, eval_dir: Path,
) -> Dict[str, Any]:
    """Probe pulchra/obabel/vina and decide skip flags. Missing tools
    DOWNGRADE the flow (skip the affected stage) rather than abort, so any
    system runs out-of-the-box. Explicit --skip-* always win."""
    cfg_path: Optional[Path] = None
    if args.external_tools_config:
        cp = Path(args.external_tools_config)
        cfg_path = cp if cp.is_absolute() else project_root / cp
        if not cfg_path.is_file():
            cfg_path = None
    # probe-only: never fail here (require_* = False); we decide below.
    pf = run_external_tools_preflight(
        pulchra_bin=args.pulchra_bin, obabel_bin=args.obabel_bin,
        vina_bin=args.vina_bin, config_path=cfg_path,
        require_pulchra=False, require_docking=False,
    )
    eval_dir.mkdir(parents=True, exist_ok=True)
    import json
    (eval_dir / "external_tools_preflight.json").write_text(
        json.dumps(pf, indent=2), encoding="utf-8",
    )

    skip_recon = bool(args.skip_reconstruction)
    skip_dock = bool(args.skip_docking)
    notes: List[str] = []

    if not pf["pulchra"]["found"] and not skip_recon:
        skip_recon = True
        notes.append("PULCHRA not found -> skipping reconstruction (and docking)")
    if skip_recon:
        skip_dock = True  # docking needs the reconstructed all-atom pose
    if not skip_dock:
        if not pf["obabel"]["found"]:
            skip_dock = True
            notes.append("OpenBabel not found -> skipping docking")
        elif not pf["vina"]["found"]:
            skip_dock = True
            notes.append("Vina not found -> skipping docking")

    return {
        "preflight": pf,
        "skip_reconstruction": skip_recon,
        "skip_docking": skip_dock,
        "notes": notes,
        "pulchra_bin": pf["pulchra"]["path"] if pf["pulchra"]["found"] else None,
        "obabel_bin": pf["obabel"]["path"] if pf["obabel"]["found"] else None,
        "vina_bin": pf["vina"]["path"] if pf["vina"]["found"] else None,
    }


# ---------------------------------------------------------------------- #
# stage 4: validation                                                    #
# ---------------------------------------------------------------------- #

def run_validation(
    run_dir: Path, eval_dir: Path, tasks: List[Any],
) -> Dict[str, int]:
    """Run FinalPipelineValidator over each sampled task_dir and write a
    per-task validation_summary.json + validation_report.md under
    eval_dir/<task_id>/validation/."""
    validator = FinalPipelineValidator()
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0, "ERROR": 0}
    for t in tasks:
        task_dir = run_dir / t.task_id
        if not task_dir.is_dir():
            continue
        out = eval_dir / t.task_id / "validation"
        try:
            res = validator.validate_task(task_dir, task_metadata=dict(t.metadata))
            write_validation_summary(out, res)
            write_validation_report(out, res)
            if res.failures > 0:
                counts["FAIL"] += 1
            elif res.warnings > 0:
                counts["WARN"] += 1
            else:
                counts["PASS"] += 1
        except Exception as e:
            counts["ERROR"] += 1
            out.mkdir(parents=True, exist_ok=True)
            (out / "validation_error.txt").write_text(repr(e), encoding="utf-8")
    return counts


# ---------------------------------------------------------------------- #
# main                                                                   #
# ---------------------------------------------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(_HERE)

    output_root = (
        Path(args.output_root) if Path(args.output_root).is_absolute()
        else project_root / args.output_root
    )
    run_dir = output_root / args.run_name
    # Downstream eval (oracle/reconstruction/docking) and validation write
    # INTO the same per-task run_dir so each task_dir is self-contained and
    # the validator (which expects one task_dir holding every stage) finds
    # the complete picture.
    eval_dir = run_dir

    print("=" * 78)
    print("UNIFIED PIPELINE — sampling -> reconstruction -> oracle -> "
          "docking -> validation")
    print("=" * 78)

    tasks_csv, structure_root = resolve_system(args, project_root, run_dir)

    # ---- stage 1: sampling (A-G) -------------------------------------
    tasks: List[Any]
    if args.skip_sampling:
        from ras_folding.kras.task_loader import load_kras_tasks
        print(f"\n[stage 1] SKIPPED (reusing run_dir = {run_dir})")
        if not run_dir.is_dir():
            raise SystemExit(
                f"--skip-sampling set but run_dir does not exist: {run_dir}"
            )
        tasks, _ = load_kras_tasks(
            tasks_csv, pdb_dir=structure_root, flank_size=1,
            skip_missing_pdb=True,
        )
    else:
        print("\n[stage 1] SAMPLING (stages A-G)")
        run_dir, tasks = run_sampling(
            args, project_root, tasks_csv, structure_root,
        )

    if args.skip_eval:
        print("\n[stage 2-4] SKIPPED (--skip-eval): sampling-only run.")
        print(f"\nDONE. sampling outputs under {run_dir}")
        return 0

    # ---- stage 2: tool preflight + skip decisions --------------------
    print("\n[stage 2] external-tool preflight (reconstruction / docking)")
    tools = resolve_downstream_tools(args, project_root, eval_dir)
    for n in tools["notes"]:
        print(f"  NOTE: {n}")
    print(f"  skip_reconstruction = {tools['skip_reconstruction']}")
    print(f"  skip_docking        = {tools['skip_docking']}")

    # ---- stage 3: downstream eval (reconstruct + oracle + docking) ---
    print("\n[stage 3] DOWNSTREAM EVAL (reconstruction + oracle + docking)")
    res = rode.run_evaluation(
        run_dir=run_dir, tasks_csv=tasks_csv, structure_root=structure_root,
        output_dir=eval_dir, max_tasks=args.max_tasks,
        repeats=int(args.repeats), exhaustiveness=int(args.exhaustiveness),
        num_modes=int(args.num_modes), temperature_k=float(args.temperature_k),
        pulchra_bin=tools["pulchra_bin"], obabel_bin=tools["obabel_bin"],
        vina_bin=tools["vina_bin"], overwrite=bool(args.overwrite),
        fail_fast=bool(args.fail_fast),
        skip_docking=tools["skip_docking"],
        skip_reconstruction=tools["skip_reconstruction"],
        use_kabsch_rmsd=bool(args.use_kabsch_rmsd),
        project_root=project_root,
    )
    print(f"  eval: n_tasks={res['n_tasks']} n_failed={res['n_failed']}")
    print(f"  eval outputs: {eval_dir}")

    # ---- stage 4: validation -----------------------------------------
    val_counts = None
    if not args.no_validation:
        print("\n[stage 4] VALIDATION (PASS / WARN / FAIL)")
        val_counts = run_validation(run_dir, eval_dir, tasks)
        print(f"  validation: {val_counts}")

    # ---- final summary -----------------------------------------------
    print("\n" + "=" * 78)
    print("PIPELINE COMPLETE")
    print(f"  run_dir            = {run_dir}  (self-contained: sampling + eval)")
    print(f"  eval summary       = {run_dir / 'oracle_docking_summary.csv'}")
    if val_counts is not None:
        print(f"  validation         = {val_counts}")
    if tools["skip_docking"]:
        print("  NOTE: docking was skipped (install Vina + OpenBabel and "
              "re-run without --skip-docking to dock).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
