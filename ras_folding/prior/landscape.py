# Author: Yuqi Zhang
"""V2 landscape file writer.

Writes the per-task `landscape/` directory used for landscape
reconstruction analysis. Critical: any oracle_anchor row is included as
a control reference but NOT counted toward generated metrics.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# Schema for landscape_candidates.csv
LANDSCAPE_CANDIDATE_FIELDS = [
    "task_id", "case_id",
    "mutation_group", "ligand_family", "pocket_module",
    "candidate_id", "source", "is_oracle_anchor",
    "coords_index", "bitstring", "count", "weight",
    "prior_mode", "prior_score_mean",
    "env_score_mean", "corridor_score_mean",
    "hard_clash_count", "soft_clash_count",
    "ligand_min_distance", "ligand_mean_distance",
    "endpoint_error", "rg",
    "kabsch_rmsd", "nokabsch_rmsd",
    "basin_id",
    "docking_best_affinity", "docking_mean_affinity", "docking_status",
]


def write_landscape(
    out_dir: Path,
    *,
    task_id: str,
    case_id: str,
    mutation_group: Optional[str],
    ligand_family: Optional[str],
    pocket_module: Optional[str],
    candidate_rows: List[Dict],
    coords: np.ndarray,                   # (N_total, n_res, 3) including anchor
    basin_summary_rows: List[Dict],
    landscape_summary: Dict,
    docking_landscape_rows: Optional[List[Dict]] = None,
) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. landscape_candidates.csv
    cand_path = out_dir / "landscape_candidates.csv"
    with cand_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LANDSCAPE_CANDIDATE_FIELDS)
        w.writeheader()
        for row in candidate_rows:
            r = {k: row.get(k, "") for k in LANDSCAPE_CANDIDATE_FIELDS}
            r["task_id"] = task_id
            r["case_id"] = case_id
            r["mutation_group"] = mutation_group or ""
            r["ligand_family"] = ligand_family or ""
            r["pocket_module"] = pocket_module or ""
            w.writerow(r)

    # 2. landscape_coords.npz
    coords_path = out_dir / "landscape_coords.npz"
    np.savez_compressed(coords_path, coords=coords.astype(np.float32))

    # 3. basin_summary.csv
    basin_path = out_dir / "basin_summary.csv"
    if basin_summary_rows:
        keys = list(basin_summary_rows[0].keys())
        with basin_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in basin_summary_rows:
                w.writerow(r)
    else:
        basin_path.write_text("basin_id\n")

    # 4. projection.csv (PCA-style placeholder; uses coords centroid for now)
    proj_path = out_dir / "projection.csv"
    with proj_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["candidate_id", "x_proj", "y_proj"])
        # Simple PCA on flattened coords if enough samples
        if coords.shape[0] >= 3:
            try:
                X = coords.reshape(coords.shape[0], -1)
                X = X - X.mean(axis=0, keepdims=True)
                # SVD-based 2-D projection
                U, S, Vt = np.linalg.svd(X, full_matrices=False)
                Z = X @ Vt[:2].T  # (N, 2)
                for i, row in enumerate(candidate_rows):
                    if i >= Z.shape[0]:
                        break
                    cid = row.get("candidate_id", f"cand_{i:06d}")
                    w.writerow([cid, float(Z[i, 0]), float(Z[i, 1])])
            except np.linalg.LinAlgError:
                pass

    # 5. landscape_summary.json
    summary_path = out_dir / "landscape_summary.json"
    summary_path.write_text(json.dumps(landscape_summary, indent=2))

    # 6. docking_landscape.csv (optional)
    if docking_landscape_rows:
        dl = out_dir / "docking_landscape.csv"
        keys = list(docking_landscape_rows[0].keys())
        with dl.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in docking_landscape_rows:
                w.writerow(r)

    # 7. mutation_ligand_context.json
    ctx = out_dir / "mutation_ligand_context.json"
    ctx.write_text(json.dumps({
        "task_id": task_id,
        "case_id": case_id,
        "mutation_group": mutation_group,
        "ligand_family": ligand_family,
        "pocket_module": pocket_module,
    }, indent=2))

    return {
        "candidates": cand_path,
        "coords": coords_path,
        "basin_summary": basin_path,
        "projection": proj_path,
        "summary": summary_path,
        "context": ctx,
    }


__all__ = ["write_landscape", "LANDSCAPE_CANDIDATE_FIELDS"]
