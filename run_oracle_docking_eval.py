# Author: Yuqi Zhang
"""Oracle + reconstruction + Vina docking evaluation.

Reads a finished KRAS full-batch run, computes the oracle-best (RMSD-min)
candidate per task, rebuilds both the predicted top-1 AND the oracle-best
into all-atom receptors via PULCHRA, then runs N Vina docking repeats per
receptor against the native ligand box.

This script is GLUE around oracle_eval, ras_folding.reconstruct, and
docking_eval. Everything is opt-in via flags — by default the script
produces oracle outputs but only attempts reconstruction + docking when
the corresponding binaries are available; missing binaries result in
``status=failed`` for that task with a clear error message.

Run from the project root:

    python run_oracle_docking_eval.py \
        --run-dir logs/kras_full_batch/full_aer_v1 \
        --tasks inputs/kras_tasks.csv \
        --structure-root kras_select_systems

Outputs land under
``logs/oracle_docking_eval/<run_name>/<task_id>/`` (per-task) and
``logs/oracle_docking_eval/<run_name>/`` (global).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

from docking_eval.pipeline import DockingEvaluationPipeline
from docking_eval.types import DockingInput
from docking_eval.pdbqt import OpenBabelPDBQTPreparer
from docking_eval.vina_runner import VinaRunner

from oracle_eval.candidate_reader import CandidateReader
from oracle_eval.rmsd_oracle import RMSDOracleSelector

from ras_folding.external_tools import run_external_tools_preflight
from ras_folding.kras.task_loader import load_kras_tasks
from ras_folding.reconstruct.pipeline import FullAtomReconstructionPipeline
from ras_folding.reconstruct.pulchra_adapter import PulchraAdapter
from ras_folding.reconstruct.types import ReconstructionInput


# ---------------------------------------------------------------------- #
def _read_predicted_top1_rmsd(task_dir: Path) -> Optional[float]:
    p = task_dir / "analysis" / "structure_analysis.json"
    if p.is_file():
        try:
            d = json.loads(p.read_text())
            t1 = d.get("top1") or {}
            v = t1.get("rmsd_to_reference")
            if v is not None:
                return float(v)
        except Exception:
            pass
    p = task_dir / "landscape" / "landscape_summary.json"
    if p.is_file():
        try:
            d = json.loads(p.read_text())
            f = d.get("funnel") or {}
            v = f.get("top1_rmsd")
            if v is not None:
                return float(v)
        except Exception:
            pass
    return None


def _structure_root_for(args, project_root: Path) -> Path:
    p = Path(args.structure_root)
    return p if p.is_absolute() else (project_root / p)


# ---------------------------------------------------------------------- #
# per-task driver                                                        #
# ---------------------------------------------------------------------- #

def evaluate_task(
    *,
    task,
    task_run_dir: Path,
    output_dir: Path,
    project_root: Path,
    structure_root: Path,
    repeats: int,
    exhaustiveness: int,
    num_modes: int,
    temperature_k: float,
    pulchra_bin: Optional[str],
    obabel_bin: Optional[str],
    vina_bin: Optional[str],
    skip_docking: bool,
    skip_reconstruction: bool,
    use_kabsch_rmsd: bool,
    docking_pipeline_factory: Callable[[], DockingEvaluationPipeline] = None,
    reconstruction_factory: Callable[[], FullAtomReconstructionPipeline] = None,
    oracle_reader: Optional[CandidateReader] = None,
) -> Dict[str, Any]:
    """Evaluate one task. Always returns a payload dict (never raises);
    failures are recorded under ``status`` / ``error`` per stage."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "task_id": task.task_id,
        "n_residues": task.encoder_inputs.n_residues,
        "ref_pdb": task.metadata.get("ref_pdb"),
        "chain_id": task.metadata.get("chain_id"),
        "start_resi": task.metadata.get("start_resi"),
        "end_resi": task.metadata.get("end_resi"),
        "ligand_resname": task.metadata.get("ligand_resname"),
        "predicted_top1_rmsd": _read_predicted_top1_rmsd(task_run_dir),
    }
    warnings: List[str] = []

    reader = oracle_reader or CandidateReader()
    candidates = reader.read_task_candidates(task_run_dir)
    payload["n_oracle_candidates"] = len(candidates)

    # ---- 1) oracle-best ------------------------------------------------
    selector = RMSDOracleSelector(use_kabsch=bool(use_kabsch_rmsd))
    oracle_dir = output_dir / "oracle"
    o_res = selector.select_best(
        task_id=task.task_id,
        candidates=candidates,
        reference_coords=task.reference_coords,
        output_dir=oracle_dir,
        sequence=task.sequence,
    )
    payload["oracle"] = {
        "best_rmsd": o_res.best_rmsd,
        "best_source": (
            o_res.best_candidate.source
            if o_res.best_candidate is not None else None
        ),
        "best_is_dense": (
            o_res.best_candidate.is_dense
            if o_res.best_candidate is not None else None
        ),
        "best_full_energy": (
            o_res.best_candidate.full_energy
            if o_res.best_candidate is not None else None
        ),
        "best_refined_score": (
            o_res.best_candidate.refined_score
            if o_res.best_candidate is not None else None
        ),
        "n_candidates_considered": o_res.n_candidates_considered,
        "reference_available": o_res.reference_available,
        "output_pdb": (None if o_res.output_pdb is None else str(o_res.output_pdb)),
    }

    # ---- 2) reconstruction --------------------------------------------
    predicted_top1_pdb = task_run_dir / "postprocess" / "top1_ca.pdb"
    ref_pdb_name = task.metadata.get("ref_pdb")
    if ref_pdb_name is None:
        raise FileNotFoundError(
            f"task {task.task_id}: ref_pdb missing in metadata"
        )
    structure_root_pdb = Path(structure_root) / ref_pdb_name
    if not structure_root_pdb.is_file():
        raise FileNotFoundError(
            f"task {task.task_id}: reference PDB not found at {structure_root_pdb}"
        )
    payload["reconstruction"] = {"predicted": None, "oracle": None}
    payload["docking"] = {"predicted_top1": None, "oracle_best": None}

    if skip_reconstruction:
        warnings.append("reconstruction skipped by flag")
    else:
        recon_factory = reconstruction_factory or _default_recon_factory(pulchra_bin)
        try:
            recon = recon_factory()
        except Exception as e:
            recon = None
            warnings.append(f"reconstruction unavailable: {e!r}")

        if recon is not None:
            # predicted top-1
            if predicted_top1_pdb.is_file():
                pr = recon.reconstruct(
                    inp=ReconstructionInput(
                        task_id=task.task_id,
                        predicted_ca_pdb=predicted_top1_pdb,
                        reference_pdb=structure_root_pdb,
                        chain_id=task.metadata.get("chain_id") or "A",
                        start_resi=int(task.metadata.get("start_resi") or 1),
                        end_resi=int(task.metadata.get("end_resi") or task.encoder_inputs.n_residues),
                        sequence=task.sequence,
                    ),
                    output_dir=output_dir / "predicted_top1",
                    ligand_resname=task.metadata.get("ligand_resname"),
                )
                payload["reconstruction"]["predicted"] = {
                    "status": pr.status,
                    "error": pr.error,
                    "embedded_receptor_pdb": (
                        None if pr.embedded_receptor_pdb is None
                        else str(pr.embedded_receptor_pdb)
                    ),
                }
            else:
                payload["reconstruction"]["predicted"] = {
                    "status": "skipped",
                    "error": "predicted_top1_ca.pdb not found",
                }

            # oracle best
            if o_res.output_pdb is not None and o_res.output_pdb.is_file():
                or_ = recon.reconstruct(
                    inp=ReconstructionInput(
                        task_id=task.task_id,
                        predicted_ca_pdb=o_res.output_pdb,
                        reference_pdb=structure_root_pdb,
                        chain_id=task.metadata.get("chain_id") or "A",
                        start_resi=int(task.metadata.get("start_resi") or 1),
                        end_resi=int(task.metadata.get("end_resi") or task.encoder_inputs.n_residues),
                        sequence=task.sequence,
                    ),
                    output_dir=output_dir / "oracle_best",
                    ligand_resname=task.metadata.get("ligand_resname"),
                )
                payload["reconstruction"]["oracle"] = {
                    "status": or_.status,
                    "error": or_.error,
                    "embedded_receptor_pdb": (
                        None if or_.embedded_receptor_pdb is None
                        else str(or_.embedded_receptor_pdb)
                    ),
                }
            else:
                payload["reconstruction"]["oracle"] = {
                    "status": "skipped",
                    "error": "oracle_best_ca.pdb missing (no candidates / no reference)",
                }

    # ---- 3) docking ----------------------------------------------------
    if skip_docking:
        warnings.append("docking skipped by flag")
    else:
        dock_factory = docking_pipeline_factory or _default_docking_factory(
            obabel_bin=obabel_bin, vina_bin=vina_bin,
            repeats=int(repeats),
            exhaustiveness=int(exhaustiveness),
            num_modes=int(num_modes),
            temperature_k=float(temperature_k),
        )
        try:
            dock = dock_factory()
        except Exception as e:
            dock = None
            warnings.append(f"docking unavailable: {e!r}")

        if dock is not None:
            # Pocket-center proxy = centroid of reference fragment CAs.
            # Used to disambiguate when the reference PDB has multiple
            # residue copies of the same ligand_resname (e.g. 7RT1's
            # 7L8 A 203 + 7L8 A 204). If reference_coords is unavailable,
            # ligand extraction falls back to the first PDB copy and
            # records a warning.
            pocket_center = None
            if (
                getattr(task, "reference_coords", None) is not None
                and task.reference_coords.shape[0] > 0
            ):
                pocket_center = (
                    np.asarray(task.reference_coords, dtype=np.float64)
                    .mean(axis=0)
                )

            # predicted top-1
            recon_pred = payload["reconstruction"]["predicted"]
            if recon_pred and recon_pred.get("status") == "done":
                inp = DockingInput(
                    task_id=task.task_id,
                    receptor_pdb=Path(recon_pred["embedded_receptor_pdb"]),
                    ligand_source_pdb=structure_root_pdb,
                    ligand_resname=task.metadata.get("ligand_resname") or "",
                    output_dir=output_dir,
                    repeats=int(repeats),
                    exhaustiveness=int(exhaustiveness),
                    num_modes=int(num_modes),
                    pocket_center=pocket_center,
                    ligand_selection_mode="nearest_to_pocket_center",
                )
                d_pred = dock.dock(inp, "predicted_top1")
                payload["docking"]["predicted_top1"] = _docking_brief(d_pred)
            else:
                payload["docking"]["predicted_top1"] = {
                    "status": "skipped",
                    "error": "no embedded_receptor_pdb for predicted_top1",
                }

            # oracle best
            recon_oracle = payload["reconstruction"]["oracle"]
            if recon_oracle and recon_oracle.get("status") == "done":
                inp = DockingInput(
                    task_id=task.task_id,
                    receptor_pdb=Path(recon_oracle["embedded_receptor_pdb"]),
                    ligand_source_pdb=structure_root_pdb,
                    ligand_resname=task.metadata.get("ligand_resname") or "",
                    output_dir=output_dir,
                    repeats=int(repeats),
                    exhaustiveness=int(exhaustiveness),
                    num_modes=int(num_modes),
                    pocket_center=pocket_center,
                    ligand_selection_mode="nearest_to_pocket_center",
                )
                d_oracle = dock.dock(inp, "oracle_best")
                payload["docking"]["oracle_best"] = _docking_brief(d_oracle)
            else:
                payload["docking"]["oracle_best"] = {
                    "status": "skipped",
                    "error": "no embedded_receptor_pdb for oracle_best",
                }

    payload["warnings"] = warnings
    (output_dir / "task_eval_summary.json").write_text(
        json.dumps(payload, indent=2, default=_jdef), encoding="utf-8",
    )
    return payload


def _structure_root_for_task(project_root: Path, ref_pdb: Optional[str]) -> Path:
    if not ref_pdb:
        raise FileNotFoundError("ref_pdb missing in task metadata")
    cand = project_root / "kras_select_systems" / ref_pdb
    return cand


def _docking_brief(d) -> Dict[str, Any]:
    return {
        "status": d.status,
        "error": d.error,
        "affinities_kcal_mol": d.affinities_kcal_mol,
        "estimated_kd_m": d.estimated_kd_m,
        "mean_affinity_kcal_mol": d.mean_affinity_kcal_mol,
        "std_affinity_kcal_mol": d.std_affinity_kcal_mol,
        "mean_kd_m": d.mean_kd_m,
        "std_kd_m": d.std_kd_m,
        "best_affinity_kcal_mol": d.best_affinity_kcal_mol,
    }


def _default_recon_factory(pulchra_bin: Optional[str]):
    def _make():
        return FullAtomReconstructionPipeline(
            pulchra_adapter=PulchraAdapter(pulchra_bin=pulchra_bin),
        )
    return _make


def _default_docking_factory(
    *, obabel_bin, vina_bin, repeats, exhaustiveness, num_modes, temperature_k,
):
    def _make():
        return DockingEvaluationPipeline(
            pdbqt_preparer=OpenBabelPDBQTPreparer(obabel_bin=obabel_bin),
            vina_runner=VinaRunner(vina_bin=vina_bin),
            repeats=repeats,
            exhaustiveness=exhaustiveness,
            num_modes=num_modes,
            temperature_k=temperature_k,
        )
    return _make


def _jdef(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


# ---------------------------------------------------------------------- #
# CLI driver                                                             #
# ---------------------------------------------------------------------- #

def run_evaluation(
    *,
    run_dir: Path,
    tasks_csv: Path,
    structure_root: Path,
    output_dir: Path,
    max_tasks: Optional[int] = None,
    task_ids: Optional[List[str]] = None,
    repeats: int = 5,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    temperature_k: float = 298.15,
    pulchra_bin: Optional[str] = None,
    obabel_bin: Optional[str] = None,
    vina_bin: Optional[str] = None,
    overwrite: bool = False,
    fail_fast: bool = False,
    skip_docking: bool = False,
    skip_reconstruction: bool = False,
    use_kabsch_rmsd: bool = False,
    docking_pipeline_factory: Callable[[], DockingEvaluationPipeline] = None,
    reconstruction_factory: Callable[[], FullAtomReconstructionPipeline] = None,
    oracle_reader: Optional[CandidateReader] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    project_root = project_root or Path(_HERE)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks, schema = load_kras_tasks(
        tasks_csv, pdb_dir=structure_root, flank_size=1, skip_missing_pdb=True,
    )
    if task_ids:
        wanted = set(task_ids)
        tasks = [t for t in tasks if t.task_id in wanted]
    if max_tasks is not None:
        tasks = tasks[: int(max_tasks)]

    rows: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    for t in tasks:
        task_run_dir = run_dir / t.task_id
        task_eval_dir = output_dir / t.task_id
        if (task_eval_dir / "task_eval_summary.json").exists() and not overwrite:
            try:
                rows.append(json.loads(
                    (task_eval_dir / "task_eval_summary.json").read_text()
                ))
            except Exception:
                pass
            continue
        if not task_run_dir.is_dir():
            failed.append({
                "task_id": t.task_id,
                "error": f"missing run dir: {task_run_dir}",
            })
            continue
        try:
            payload = evaluate_task(
                task=t,
                task_run_dir=task_run_dir,
                output_dir=task_eval_dir,
                project_root=project_root,
                structure_root=structure_root,
                repeats=repeats,
                exhaustiveness=exhaustiveness,
                num_modes=num_modes,
                temperature_k=temperature_k,
                pulchra_bin=pulchra_bin,
                obabel_bin=obabel_bin,
                vina_bin=vina_bin,
                skip_docking=skip_docking,
                skip_reconstruction=skip_reconstruction,
                use_kabsch_rmsd=use_kabsch_rmsd,
                docking_pipeline_factory=docking_pipeline_factory,
                reconstruction_factory=reconstruction_factory,
                oracle_reader=oracle_reader,
            )
            rows.append(payload)
        except Exception as e:
            err = traceback.format_exc()
            (task_eval_dir).mkdir(parents=True, exist_ok=True)
            (task_eval_dir / "ERROR.txt").write_text(err)
            failed.append({"task_id": t.task_id, "error": repr(e)})
            if fail_fast:
                break

    _write_global_outputs(output_dir, rows, failed)
    return {
        "n_tasks": len(rows),
        "n_failed": len(failed),
        "output_dir": str(output_dir),
        "rows": rows,
        "failed": failed,
    }


def _write_global_outputs(
    output_dir: Path,
    rows: List[Dict[str, Any]],
    failed: List[Dict[str, Any]],
) -> None:
    csv_path = output_dir / "oracle_docking_summary.csv"
    cols = [
        "task_id", "n_residues", "ref_pdb", "chain_id",
        "start_resi", "end_resi", "ligand_resname",
        "predicted_top1_rmsd",
        "oracle_best_rmsd", "oracle_best_source", "oracle_best_is_dense",
        "oracle_best_full_energy", "oracle_best_refined_score",
        "predicted_reconstruction_status", "oracle_reconstruction_status",
        "predicted_docking_status", "oracle_docking_status",
        "predicted_mean_affinity_kcal_mol", "predicted_std_affinity_kcal_mol",
        "predicted_best_affinity_kcal_mol",
        "predicted_mean_kd_m", "predicted_std_kd_m",
        "oracle_mean_affinity_kcal_mol", "oracle_std_affinity_kcal_mol",
        "oracle_best_affinity_kcal_mol",
        "oracle_mean_kd_m", "oracle_std_kd_m",
        "delta_rmsd_pred_minus_oracle",
        "delta_affinity_pred_minus_oracle",
        "kd_ratio_pred_over_oracle",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            o = r.get("oracle") or {}
            recon = r.get("reconstruction") or {}
            dock = r.get("docking") or {}
            pred_d = dock.get("predicted_top1") or {}
            oracle_d = dock.get("oracle_best") or {}
            row = {
                "task_id": r.get("task_id"),
                "n_residues": r.get("n_residues"),
                "ref_pdb": r.get("ref_pdb"),
                "chain_id": r.get("chain_id"),
                "start_resi": r.get("start_resi"),
                "end_resi": r.get("end_resi"),
                "ligand_resname": r.get("ligand_resname"),
                "predicted_top1_rmsd": _fmt(r.get("predicted_top1_rmsd")),
                "oracle_best_rmsd": _fmt(o.get("best_rmsd")),
                "oracle_best_source": o.get("best_source"),
                "oracle_best_is_dense": o.get("best_is_dense"),
                "oracle_best_full_energy": _fmt(o.get("best_full_energy")),
                "oracle_best_refined_score": _fmt(o.get("best_refined_score")),
                "predicted_reconstruction_status": (
                    (recon.get("predicted") or {}).get("status")
                ),
                "oracle_reconstruction_status": (
                    (recon.get("oracle") or {}).get("status")
                ),
                "predicted_docking_status": pred_d.get("status"),
                "oracle_docking_status": oracle_d.get("status"),
                "predicted_mean_affinity_kcal_mol": _fmt(
                    pred_d.get("mean_affinity_kcal_mol")
                ),
                "predicted_std_affinity_kcal_mol": _fmt(
                    pred_d.get("std_affinity_kcal_mol")
                ),
                "predicted_best_affinity_kcal_mol": _fmt(
                    pred_d.get("best_affinity_kcal_mol")
                ),
                "predicted_mean_kd_m": _fmt_sci(pred_d.get("mean_kd_m")),
                "predicted_std_kd_m": _fmt_sci(pred_d.get("std_kd_m")),
                "oracle_mean_affinity_kcal_mol": _fmt(
                    oracle_d.get("mean_affinity_kcal_mol")
                ),
                "oracle_std_affinity_kcal_mol": _fmt(
                    oracle_d.get("std_affinity_kcal_mol")
                ),
                "oracle_best_affinity_kcal_mol": _fmt(
                    oracle_d.get("best_affinity_kcal_mol")
                ),
                "oracle_mean_kd_m": _fmt_sci(oracle_d.get("mean_kd_m")),
                "oracle_std_kd_m": _fmt_sci(oracle_d.get("std_kd_m")),
                "delta_rmsd_pred_minus_oracle": _fmt(_safe_diff(
                    r.get("predicted_top1_rmsd"), o.get("best_rmsd"),
                )),
                "delta_affinity_pred_minus_oracle": _fmt(_safe_diff(
                    pred_d.get("mean_affinity_kcal_mol"),
                    oracle_d.get("mean_affinity_kcal_mol"),
                )),
                "kd_ratio_pred_over_oracle": _fmt_sci(_safe_ratio(
                    pred_d.get("mean_kd_m"), oracle_d.get("mean_kd_m"),
                )),
                "error": r.get("error"),
            }
            w.writerow(row)

    (output_dir / "oracle_docking_summary.json").write_text(
        json.dumps(rows, indent=2, default=_jdef), encoding="utf-8",
    )

    # failed_tasks.csv
    if failed:
        with (output_dir / "failed_tasks.csv").open(
            "w", encoding="utf-8", newline="",
        ) as fh:
            w = csv.DictWriter(fh, fieldnames=["task_id", "error"])
            w.writeheader()
            for f in failed:
                w.writerow(f)

    # markdown report (compact)
    md = ["# Oracle + docking evaluation", ""]
    md.append(f"- n_tasks: {len(rows)}")
    md.append(f"- n_failed: {len(failed)}")
    md.append("")
    md.append("| task_id | predicted_top1_rmsd | oracle_best_rmsd | "
              "predicted_mean_kd_m | oracle_mean_kd_m | kd_ratio_pred_over_oracle |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        o = r.get("oracle") or {}
        dock = r.get("docking") or {}
        pred_d = dock.get("predicted_top1") or {}
        oracle_d = dock.get("oracle_best") or {}
        md.append(
            f"| {r.get('task_id')} | "
            f"{_fmt(r.get('predicted_top1_rmsd'))} | "
            f"{_fmt(o.get('best_rmsd'))} | "
            f"{_fmt_sci(pred_d.get('mean_kd_m'))} | "
            f"{_fmt_sci(oracle_d.get('mean_kd_m'))} | "
            f"{_fmt_sci(_safe_ratio(pred_d.get('mean_kd_m'), oracle_d.get('mean_kd_m')))} |"
        )
    md.append("")
    md.extend(_render_multi_copy_ligand_section(output_dir, rows))
    (output_dir / "oracle_docking_report.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8",
    )


# ---------------------------------------------------------------------- #
# Multi-copy ligand handling section                                     #
# ---------------------------------------------------------------------- #

def _read_extraction_summary(p: Path) -> Optional[Dict[str, Any]]:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _collect_multi_copy_ligand_info(
    output_dir: Path,
    rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """For each row, locate ``ligand_extraction_summary.json`` under
    ``<task>/predicted_top1/docking/`` and ``<task>/oracle_best/docking/``
    and gather rows where ``n_ligand_copies_found > 1``. Returns one
    entry per (task, receptor_label) pair that had multi-copy ligands.

    Read-only; never raises if files are missing.
    """
    info: List[Dict[str, Any]] = []
    for r in rows:
        task_id = r.get("task_id")
        if not task_id:
            continue
        task_dir = output_dir / str(task_id)
        for label in ("predicted_top1", "oracle_best"):
            sm = _read_extraction_summary(
                task_dir / label / "docking" / "ligand_extraction_summary.json"
            )
            if sm is None:
                continue
            n_copies = int(sm.get("n_ligand_copies_found") or 0)
            if n_copies <= 1:
                continue
            dock = (r.get("docking") or {}).get(label) or {}
            handled = dock.get("status") == "done"
            info.append({
                "task_id": task_id,
                "receptor_label": label,
                "ref_pdb": r.get("ref_pdb"),
                "ligand_resname": sm.get("ligand_resname"),
                "selected_chain_id": sm.get("selected_chain_id"),
                "selected_resseq": sm.get("selected_resseq"),
                "selected_icode": sm.get("selected_icode"),
                "selected_distance_to_pocket": sm.get(
                    "selected_distance_to_pocket"
                ),
                "ligand_selection_mode": sm.get("ligand_selection_mode"),
                "n_ligand_copies_found": n_copies,
                "candidates": sm.get("candidates") or [],
                "warnings": sm.get("warnings") or [],
                "handled": handled,
            })
    return info


def _render_multi_copy_ligand_section(
    output_dir: Path,
    rows: List[Dict[str, Any]],
) -> List[str]:
    """Return the markdown lines for the "Multi-copy ligand handling"
    section. Empty list if no task has multi-copy ligands."""
    info = _collect_multi_copy_ligand_info(output_dir, rows)
    if not info:
        return []
    lines: List[str] = []
    lines.append("## Multi-copy ligand handling")
    lines.append("")
    lines.append(
        "Some reference PDBs contain more than one residue copy of "
        "the docking-target ligand resname. The pipeline selects the "
        "single copy whose centroid is **nearest to the fragment CA "
        "centroid** (`ligand_selection_mode=nearest_to_pocket_center`) "
        "and writes the selection to "
        "`<task>/<label>/docking/ligand_extraction_summary.json`. "
        "Multi-copy ligands are NOT failures — they are explicitly "
        "handled. Where the rule selects a non-canonical copy, this "
        "section records it so manuscript / report writing can "
        "annotate the choice."
    )
    lines.append("")
    lines.append(
        "| task_id | label | ref_pdb | resname | chain | resseq | "
        "icode | distance(Å) | n_copies | mode | handled |"
    )
    lines.append("|---|---|---|---|---|---:|---|---:|---:|---|---|")
    for e in info:
        lines.append(
            f"| {e['task_id']} | {e['receptor_label']} | "
            f"{e['ref_pdb']} | {e['ligand_resname']} | "
            f"{e['selected_chain_id']} | {e['selected_resseq']} | "
            f"{e['selected_icode'] or ''} | "
            f"{_fmt(e['selected_distance_to_pocket'])} | "
            f"{e['n_ligand_copies_found']} | "
            f"{e['ligand_selection_mode']} | "
            f"{'yes' if e['handled'] else 'no'} |"
        )
    lines.append("")
    lines.append("### Per-task candidate copies")
    lines.append("")
    seen_keys: set = set()
    for e in info:
        key = (e["task_id"], e["receptor_label"])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        lines.append(
            f"- **{e['task_id']}** ({e['receptor_label']}) — "
            f"`{e['ref_pdb']}` `{e['ligand_resname']}`, "
            f"selected chain {e['selected_chain_id']} "
            f"resseq {e['selected_resseq']} "
            f"(distance {_fmt(e['selected_distance_to_pocket'])} Å, "
            f"mode `{e['ligand_selection_mode']}`):"
        )
        for c in e["candidates"]:
            d = c.get("distance_to_pocket_center")
            d_s = _fmt(d) if d is not None else "n/a"
            lines.append(
                f"  - chain {c.get('chain_id')} "
                f"resseq {c.get('resseq')} "
                f"icode `{c.get('icode') or ''}` "
                f"n_atoms {c.get('n_atoms')} "
                f"distance_to_pocket={d_s}"
            )
    lines.append("")
    lines.append(
        "**Note on 7RT1 / 7L8.** RCSB 7RT1 deposits the inhibitor 7L8 "
        "twice in chain A (resseq 203 and 204). Under the nearest-to-"
        "pocket-center rule, fragments whose CA centroid is closer to "
        "the resseq-204 copy (e.g. Switch-II Cterm residues 69–78, "
        "i.e. Frag-F) select 204; fragments closer to the 203 copy "
        "(P-loop, Switch-II Nterm, α3 allosteric — Frag-A/B/C) select "
        "203. Both copies are valid binding poses of 7L8 in the "
        "deposited crystal; the choice is **not** a ligand extraction "
        "error, just a deterministic geometric tie-break."
    )
    return lines


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.6f}"
    except Exception:
        return str(x)


def _fmt_sci(x: Any) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.6e}"
    except Exception:
        return str(x)


def _safe_diff(a, b):
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _safe_ratio(a, b):
    if a is None or b is None or float(b) == 0.0:
        return None
    return float(a) / float(b)


# ---------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Oracle + Vina docking evaluation of a finished KRAS run.",
    )
    p.add_argument(
        "--run-dir", type=str,
        default="logs/kras_full_batch/full_ibm_v1",
        help="Path to a completed KRAS full-batch run directory.",
    )
    p.add_argument("--tasks", type=str, default="inputs/kras_tasks.csv")
    p.add_argument("--structure-root", type=str, default="kras_select_systems")
    p.add_argument(
        "--output-dir", type=str,
        default=None,
        help="Default: logs/oracle_docking_eval/<basename of --run-dir>",
    )
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument(
        "--task-ids", type=str, default=None,
        help="Comma-separated list of task_ids to restrict evaluation to.",
    )
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--num-modes", type=int, default=9)
    p.add_argument("--temperature-k", type=float, default=298.15)
    p.add_argument("--pulchra-bin", type=str, default=None)
    p.add_argument("--obabel-bin", type=str, default=None)
    p.add_argument("--vina-bin", type=str, default=None)
    p.add_argument(
        "--external-tools-config", type=str, default=None,
        help=(
            "Optional path to external_tools.json. Templates: "
            "external_tools.example.json, "
            "external_tools.example.windows.json."
        ),
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--skip-docking", action="store_true")
    p.add_argument("--skip-reconstruction", action="store_true")
    p.add_argument("--use-kabsch-rmsd", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    project_root = Path(_HERE)

    run_dir = (
        Path(args.run_dir) if Path(args.run_dir).is_absolute()
        else project_root / args.run_dir
    )
    tasks_csv = (
        Path(args.tasks) if Path(args.tasks).is_absolute()
        else project_root / args.tasks
    )
    structure_root = (
        Path(args.structure_root) if Path(args.structure_root).is_absolute()
        else project_root / args.structure_root
    )
    if args.output_dir is None:
        out_dir = (
            project_root / "logs" / "oracle_docking_eval" / run_dir.name
        )
    else:
        out_dir = (
            Path(args.output_dir) if Path(args.output_dir).is_absolute()
            else project_root / args.output_dir
        )
    task_ids = (
        [t.strip() for t in args.task_ids.split(",") if t.strip()]
        if args.task_ids else None
    )

    print(f"[oracle-docking] run_dir       = {run_dir}")
    print(f"[oracle-docking] output_dir    = {out_dir}")
    print(f"[oracle-docking] skip_docking  = {args.skip_docking}")
    print(f"[oracle-docking] skip_reconstruction = {args.skip_reconstruction}")

    # ---- preflight BEFORE we start the batch -------------------------
    # Required tools depend on which downstream stages will actually run.
    require_pulchra = not bool(args.skip_reconstruction)
    require_docking = not bool(args.skip_docking)
    cfg_path: Optional[Path] = None
    if args.external_tools_config:
        cp = Path(args.external_tools_config)
        cfg_path = cp if cp.is_absolute() else project_root / cp
    pf = run_external_tools_preflight(
        pulchra_bin=args.pulchra_bin,
        obabel_bin=args.obabel_bin,
        vina_bin=args.vina_bin,
        config_path=cfg_path,
        require_pulchra=require_pulchra,
        require_docking=require_docking,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pf_path = out_dir / "external_tools_preflight.json"
    pf_path.write_text(json.dumps(pf, indent=2), encoding="utf-8")
    if not pf["ok"]:
        print("=" * 78)
        print("EXTERNAL TOOL PREFLIGHT FAILED — batch ABORTED.")
        print(f"  missing tools     : {pf['missing']}")
        print(f"  preflight report  : {pf_path}")
        for key in pf["missing"]:
            print(f"  --- {key} ---")
            for a in pf[key].get("attempted", []):
                print(
                    f"    - {a['source']}: {a['value']!r} [{a['status']}]"
                )
        print(
            "Fix: write external_tools.json (see external_tools.example.json), "
            "or pass --pulchra-bin/--obabel-bin/--vina-bin explicitly."
        )
        print("=" * 78)
        raise SystemExit(2)
    # Lock the resolved absolute paths into the args so adapters never
    # silently fall back to PATH.
    if pf["pulchra"]["found"]:
        args.pulchra_bin = pf["pulchra"]["path"]
    if pf["obabel"]["found"]:
        args.obabel_bin = pf["obabel"]["path"]
    if pf["vina"]["found"]:
        args.vina_bin = pf["vina"]["path"]

    res = run_evaluation(
        run_dir=run_dir,
        tasks_csv=tasks_csv,
        structure_root=structure_root,
        output_dir=out_dir,
        max_tasks=args.max_tasks,
        task_ids=task_ids,
        repeats=int(args.repeats),
        exhaustiveness=int(args.exhaustiveness),
        num_modes=int(args.num_modes),
        temperature_k=float(args.temperature_k),
        pulchra_bin=args.pulchra_bin,
        obabel_bin=args.obabel_bin,
        vina_bin=args.vina_bin,
        overwrite=bool(args.overwrite),
        fail_fast=bool(args.fail_fast),
        skip_docking=bool(args.skip_docking),
        skip_reconstruction=bool(args.skip_reconstruction),
        use_kabsch_rmsd=bool(args.use_kabsch_rmsd),
        project_root=project_root,
    )
    print(f"[oracle-docking] done: n_tasks={res['n_tasks']} n_failed={res['n_failed']}")
    print(f"[oracle-docking] output: {res['output_dir']}")


if __name__ == "__main__":
    main()
