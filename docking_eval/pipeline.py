# Author: Yuqi Zhang
"""Docking evaluation pipeline — extract ligand, prep PDBQT, run Vina N times."""
from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ---------------------------------------------------------------------- #
# Stale ERROR.txt hygiene                                                #
# ---------------------------------------------------------------------- #
#
# Per-stage ERROR.txt files are written ONLY when the stage failed. If a
# subsequent (successful) re-run leaves the old ERROR.txt around, manual
# audit / diff-style review is misled. The two helpers below let any
# stage in this module remove the stale file before retrying *and*
# re-confirm the absence after success.

def remove_stale_error_txt(stage_dir: Path) -> None:
    """Delete ``stage_dir/ERROR.txt`` if it exists. No-op otherwise."""
    p = Path(stage_dir) / "ERROR.txt"
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass


def clear_stale_error_if_success(
    stage_dir: Path,
    success_markers: Iterable[Path],
) -> bool:
    """If ERROR.txt exists in ``stage_dir`` AND every path in
    ``success_markers`` exists, remove ERROR.txt. Returns True iff a
    stale file was actually removed.

    This is the auditing equivalent of ``remove_stale_error_txt``: it
    requires positive proof that the stage succeeded before deleting
    the failure marker.
    """
    err = Path(stage_dir) / "ERROR.txt"
    if not err.exists():
        return False
    for m in success_markers:
        if not Path(m).exists():
            return False
    try:
        err.unlink()
    except OSError:
        return False
    return True

import numpy as np

from docking_eval.kd import affinity_to_kd_m
from docking_eval.ligand import (
    compute_ligand_box,
    extract_ligand_from_pdb_with_metadata,
)
from docking_eval.pdbqt import (
    OpenBabelPDBQTPreparer,
    assert_single_root_pdbqt,
    count_pdbqt_root_blocks,
)
from docking_eval.types import DockingInput, DockingRunResult
from docking_eval.vina_runner import VinaRunner


class DockingEvaluationPipeline:
    def __init__(
        self,
        pdbqt_preparer: Optional[OpenBabelPDBQTPreparer] = None,
        vina_runner: Optional[VinaRunner] = None,
        repeats: int = 5,
        exhaustiveness: int = 8,
        num_modes: int = 9,
        temperature_k: float = 298.15,
    ) -> None:
        self.preparer = pdbqt_preparer or OpenBabelPDBQTPreparer()
        self.vina = vina_runner or VinaRunner()
        self.repeats = int(repeats)
        self.exhaustiveness = int(exhaustiveness)
        self.num_modes = int(num_modes)
        self.temperature_k = float(temperature_k)

    # ------------------------------------------------------------------ #
    def dock(
        self,
        inp: DockingInput,
        receptor_label: str,
    ) -> DockingRunResult:
        out_dir = Path(inp.output_dir) / receptor_label / "docking"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Stale-state hygiene: if the previous attempt at this stage
        # failed and left ERROR.txt behind, clear it before retrying.
        # The except clause below will rewrite ERROR.txt only on
        # genuine failure of this attempt.
        remove_stale_error_txt(out_dir)
        files: Dict[str, str] = {}
        meta: Dict[str, Any] = {
            "receptor_pdb": str(inp.receptor_pdb),
            "ligand_source_pdb": str(inp.ligand_source_pdb),
            "ligand_resname": inp.ligand_resname,
            "repeats": self.repeats,
            "exhaustiveness": self.exhaustiveness,
            "num_modes": self.num_modes,
            "temperature_k": self.temperature_k,
        }

        try:
            # --- 1) extract native ligand (one residue copy only)
            ligand_native = out_dir / "ligand_native.pdb"
            extraction = extract_ligand_from_pdb_with_metadata(
                pdb_path=inp.ligand_source_pdb,
                ligand_resname=inp.ligand_resname,
                output_pdb=ligand_native,
                pocket_center=inp.pocket_center,
                ligand_selection_mode=str(inp.ligand_selection_mode),
            )
            files["ligand_native.pdb"] = str(ligand_native)
            extraction_summary = {
                "ligand_resname": extraction.ligand_resname,
                "ligand_selection_mode": extraction.selection_mode,
                "n_ligand_copies_found": extraction.n_ligand_copies_found,
                "selected_chain_id": extraction.selected_chain_id,
                "selected_resseq": extraction.selected_resseq,
                "selected_icode": extraction.selected_icode,
                "selected_distance_to_pocket": (
                    extraction.selected_distance_to_pocket
                ),
                "candidates": extraction.candidates,
                "warnings": extraction.warnings,
                "metadata": extraction.metadata,
            }
            (out_dir / "ligand_extraction_summary.json").write_text(
                json.dumps(extraction_summary, indent=2), encoding="utf-8",
            )
            files["ligand_extraction_summary.json"] = str(
                out_dir / "ligand_extraction_summary.json"
            )
            meta["ligand_extraction"] = {
                "ligand_resname": extraction.ligand_resname,
                "ligand_selection_mode": extraction.selection_mode,
                "n_ligand_copies_found": extraction.n_ligand_copies_found,
                "selected_chain_id": extraction.selected_chain_id,
                "selected_resseq": extraction.selected_resseq,
                "selected_icode": extraction.selected_icode,
                "selected_distance_to_pocket": (
                    extraction.selected_distance_to_pocket
                ),
                "warnings": extraction.warnings,
            }

            # --- 2) box
            box = compute_ligand_box(
                ligand_native,
                padding=float(inp.box_padding),
                min_box_size=float(inp.min_box_size),
            )
            (out_dir / "box.json").write_text(
                json.dumps(box, indent=2), encoding="utf-8",
            )
            files["box.json"] = str(out_dir / "box.json")
            meta["box"] = box

            # --- 3) prepare PDBQTs
            receptor_pdbqt = out_dir / "receptor.pdbqt"
            self.preparer.prepare_receptor(
                inp.receptor_pdb, receptor_pdbqt,
            )
            ligand_pdbqt = out_dir / "ligand.pdbqt"
            self.preparer.prepare_ligand(
                ligand_native, ligand_pdbqt,
            )
            files["receptor.pdbqt"] = str(receptor_pdbqt)
            files["ligand.pdbqt"] = str(ligand_pdbqt)

            # ROOT-block sanity check: catches multi-ligand merges before
            # they reach Vina (which would emit only a cryptic "Unknown
            # or inappropriate tag" parse error).
            n_root = assert_single_root_pdbqt(ligand_pdbqt)
            meta["ligand_pdbqt_root_blocks"] = n_root

            # --- 4) run Vina N times
            affinities: List[float] = []
            kds: List[float] = []
            for i in range(self.repeats):
                out_pdbqt = out_dir / f"docking_repeat_{i:02d}_out.pdbqt"
                log_path = out_dir / f"docking_repeat_{i:02d}.log"
                aff = self.vina.run_once(
                    receptor_pdbqt=receptor_pdbqt,
                    ligand_pdbqt=ligand_pdbqt,
                    box=box,
                    output_pdbqt=out_pdbqt,
                    log_path=log_path,
                    exhaustiveness=self.exhaustiveness,
                    num_modes=self.num_modes,
                    seed=int(inp.seed) + i,
                )
                affinities.append(float(aff))
                kds.append(float(affinity_to_kd_m(
                    aff, temperature_k=self.temperature_k,
                )))

            mean_aff = float(statistics.fmean(affinities)) if affinities else None
            std_aff = (
                float(statistics.pstdev(affinities))
                if len(affinities) >= 2 else 0.0 if affinities else None
            )
            mean_kd = float(statistics.fmean(kds)) if kds else None
            std_kd = (
                float(statistics.pstdev(kds))
                if len(kds) >= 2 else 0.0 if kds else None
            )
            best_aff = float(min(affinities)) if affinities else None

            # --- 5) write per-task summary + scores CSV
            self._write_scores_csv(
                out_dir / "docking_scores.csv",
                affinities, kds, base_seed=int(inp.seed),
            )
            files["docking_scores.csv"] = str(out_dir / "docking_scores.csv")
            summary = {
                "task_id": inp.task_id,
                "receptor_label": receptor_label,
                "affinities_kcal_mol": affinities,
                "estimated_kd_m": kds,
                "mean_affinity_kcal_mol": mean_aff,
                "std_affinity_kcal_mol": std_aff,
                "mean_kd_m": mean_kd,
                "std_kd_m": std_kd,
                "best_affinity_kcal_mol": best_aff,
                "ligand_extraction": meta.get("ligand_extraction"),
                "ligand_pdbqt_root_blocks": meta.get("ligand_pdbqt_root_blocks"),
                "metadata": meta,
            }
            (out_dir / "docking_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8",
            )
            files["docking_summary.json"] = str(out_dir / "docking_summary.json")

            # Defensive: with success_markers fully written, any
            # ERROR.txt sitting in the dir must be stale. Remove it.
            clear_stale_error_if_success(
                out_dir,
                [
                    out_dir / "docking_summary.json",
                    out_dir / "docking_scores.csv",
                ],
            )

            return DockingRunResult(
                task_id=inp.task_id,
                receptor_label=receptor_label,
                affinities_kcal_mol=affinities,
                estimated_kd_m=kds,
                mean_affinity_kcal_mol=mean_aff,
                std_affinity_kcal_mol=std_aff,
                mean_kd_m=mean_kd,
                std_kd_m=std_kd,
                best_affinity_kcal_mol=best_aff,
                output_files=files,
                status="done",
                metadata=meta,
            )

        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            (out_dir / "ERROR.txt").write_text(err, encoding="utf-8")
            return DockingRunResult(
                task_id=inp.task_id,
                receptor_label=receptor_label,
                affinities_kcal_mol=[],
                estimated_kd_m=[],
                mean_affinity_kcal_mol=None,
                std_affinity_kcal_mol=None,
                mean_kd_m=None,
                std_kd_m=None,
                best_affinity_kcal_mol=None,
                output_files=files,
                status="failed",
                error=err,
                metadata=meta,
            )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _write_scores_csv(
        path: Path,
        affinities: List[float],
        kds: List[float],
        base_seed: int,
    ) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["repeat_index", "seed", "affinity_kcal_mol", "estimated_kd_m"])
            for i, (a, k) in enumerate(zip(affinities, kds)):
                w.writerow([i, base_seed + i, f"{a:.6f}", f"{k:.6e}"])


__all__ = [
    "DockingEvaluationPipeline",
    "clear_stale_error_if_success",
    "remove_stale_error_txt",
]
