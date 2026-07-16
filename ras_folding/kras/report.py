# Author: Yuqi Zhang
"""Markdown report writers for per-task and global results."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _f(x, fmt="{:.4f}"):
    if x is None:
        return "None"
    try:
        return fmt.format(float(x))
    except Exception:
        return str(x)


def write_task_report(
    task_dir: Path,
    payload: Dict[str, Any],
) -> Path:
    """Write `task_dir/analysis/task_report.md`. ``payload`` is the runner's
    per-task summary dict (see KrasFullBatchRunner._build_task_summary)."""
    out_dir = task_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "task_report.md"

    task = payload.get("task", {})
    backend = payload.get("backend", {})
    per_tau = payload.get("per_tau", []) or []
    densify = payload.get("densify") or {}
    refinement = payload.get("refinement") or {}
    post = payload.get("post") or {}
    landscape = payload.get("landscape") or {}
    structure = payload.get("structure") or {}
    warnings = payload.get("warnings") or []

    lines: List[str] = []
    lines.append(f"# {task.get('task_id', 'KRAS task')}")
    lines.append("")
    lines.append("## Task")
    lines.append(f"- task_id: `{task.get('task_id')}`")
    lines.append(f"- sequence: `{task.get('sequence')}`")
    lines.append(f"- n_residues: {task.get('n_residues')}")
    lines.append(
        f"- ref_pdb: `{task.get('ref_pdb')}` "
        f"(has_native_ref={task.get('has_native_ref')})"
    )
    lines.append(
        f"- chain={task.get('chain_id')} "
        f"resi=[{task.get('start_resi')}, {task.get('end_resi')}]"
    )
    # CSV-extra grouping fields (only printed when present in CSV).
    if task.get("mutation_group"):
        lines.append(f"- Mutation group: {task.get('mutation_group')}")
    if task.get("ligand_family"):
        lines.append(f"- Ligand family: {task.get('ligand_family')}")
    if task.get("pocket_module"):
        lines.append(f"- Pocket module: {task.get('pocket_module')}")
    if task.get("analysis_role"):
        lines.append(f"- Analysis role: {task.get('analysis_role')}")
    lines.append("")

    lines.append("## Backend")
    for k, v in backend.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## Per τ")
    for t in per_tau:
        lines.append(
            f"- τ={t.get('tau')}: raw={t.get('n_raw')} "
            f"valid={t.get('n_valid')} accepted={t.get('n_accepted')} "
            f"acc_rate={_f(t.get('acceptance_rate'))} "
            f"valid_rate={_f(t.get('valid_rate'))} "
            f"fallback={t.get('fallback_triggered')} "
            f"mean_filter_acc={_f(t.get('mean_filter_energy_accepted'))} "
            f"mean_full_acc={_f(t.get('mean_full_energy_accepted'))}"
        )
    lines.append("")

    lines.append("## Densify")
    if not densify:
        lines.append("- disabled")
    else:
        for k in (
            "n_parent_selected", "n_children_generated", "n_children_kept",
            "mean_local_rmsd_to_parent", "best_energy_delta",
        ):
            lines.append(f"- {k}: {_f(densify.get(k))}")
        if "rejection_reason_counts" in densify:
            lines.append(
                f"- rejection_reason_counts: {densify['rejection_reason_counts']}"
            )
    lines.append("")

    lines.append("## Refinement")
    for k in (
        "n_eligible", "n_selected", "n_original_candidates",
        "n_dense_candidates", "n_dense_selected_in_subspace",
        "dense_fraction_in_subspace", "refined_weight_entropy",
        "dense_fraction_cap_applied", "n_dense_removed_by_cap",
        "top_k_overlap_energy_vs_refined",
    ):
        if k in refinement:
            lines.append(f"- {k}: {_f(refinement.get(k))}")
    lines.append("")

    lines.append("## Postprocess")
    for k in (
        "n_after_validity_filter", "n_after_bitstring_dedup",
        "n_after_structure_dedup", "n_basins",
        "top_basin_weight", "top_basin_weight_ratio",
        "refined_weight_entropy",
        "top1_full_energy", "top1_refined_score",
        "top1_filter_energy", "top1_bitstring", "top1_is_dense",
    ):
        if k in post:
            lines.append(f"- {k}: {_f(post.get(k))}")
    lines.append("")

    lines.append("## Landscape")
    if not landscape:
        lines.append("- disabled")
    else:
        for k in (
            "n_candidates", "n_dense", "n_original",
            "n_basins", "pca_explained_variance_ratio",
        ):
            if k in landscape:
                lines.append(f"- {k}: {landscape.get(k)}")
        if "funnel" in landscape and landscape["funnel"]:
            lines.append("- funnel:")
            for k, v in landscape["funnel"].items():
                lines.append(f"    - {k}: {_f(v)}")
        elif "spread" in landscape and landscape["spread"]:
            lines.append("- spread:")
            for k, v in landscape["spread"].items():
                lines.append(f"    - {k}: {_f(v)}")
    lines.append("")

    lines.append("## Structure analysis")
    if not structure:
        lines.append("- disabled")
    else:
        if structure.get("top1"):
            lines.append("### top1")
            for k, v in structure["top1"].items():
                lines.append(f"- {k}: {_f(v)}")
        if structure.get("top_k"):
            lines.append("### top_k")
            for k, v in structure["top_k"].items():
                lines.append(f"- {k}: {_f(v)}")
        if structure.get("basins"):
            lines.append("### basins")
            for k, v in structure["basins"].items():
                lines.append(f"- {k}: {_f(v)}")
        if structure.get("geometry_sanity"):
            lines.append("### geometry_sanity")
            for k, v in structure["geometry_sanity"].items():
                lines.append(f"- {k}: {_f(v)}")
    lines.append("")

    if "top1_pdb" in payload and payload["top1_pdb"]:
        lines.append(f"## Top-1 CA PDB\n- `{payload['top1_pdb']}`\n")
    if "basin_pdbs" in payload and payload["basin_pdbs"]:
        lines.append("## Basin representative PDBs")
        for pdb_path in payload["basin_pdbs"]:
            lines.append(f"- `{pdb_path}`")
        lines.append("")

    if warnings:
        lines.append("## Warnings")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def write_global_report(
    output_root: Path,
    payload: Dict[str, Any],
) -> Path:
    """Write `output_root/global_report.md` summarising all tasks."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    p = output_root / "global_report.md"

    lines: List[str] = []
    lines.append("# KRAS full-batch report")
    lines.append("")
    lines.append("## Run")
    for k in ("backend", "n_tasks", "n_success", "n_failed"):
        lines.append(f"- {k}: {payload.get(k)}")
    lines.append(f"- total_shots: {payload.get('total_shots')}")
    lines.append(f"- total_ibm_jobs: {payload.get('total_ibm_jobs')}")
    lines.append("")

    if payload.get("valid_rate_distribution"):
        lines.append("## Valid rate distribution")
        for k, v in payload["valid_rate_distribution"].items():
            lines.append(f"- {k}: {_f(v)}")
        lines.append("")

    if payload.get("accepted_rate_distribution"):
        lines.append("## Accepted rate distribution")
        for k, v in payload["accepted_rate_distribution"].items():
            lines.append(f"- {k}: {_f(v)}")
        lines.append("")

    if payload.get("top1_rmsd_distribution"):
        lines.append("## Top-1 RMSD distribution (where reference exists)")
        for k, v in payload["top1_rmsd_distribution"].items():
            lines.append(f"- {k}: {_f(v)}")
        lines.append("")

    if payload.get("best_rmsd_distribution"):
        lines.append("## Best top-5 RMSD distribution (where reference exists)")
        for k, v in payload["best_rmsd_distribution"].items():
            lines.append(f"- {k}: {_f(v)}")
        lines.append("")

    if payload.get("basin_count_distribution"):
        lines.append("## Basin count distribution")
        for k, v in payload["basin_count_distribution"].items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    if payload.get("failed_tasks"):
        lines.append("## Failed tasks")
        for t in payload["failed_tasks"]:
            lines.append(f"- {t.get('task_id')}: {t.get('error')}")
        lines.append("")

    p.write_text("\n".join(lines), encoding="utf-8")
    return p


__all__ = ["write_task_report", "write_global_report"]
