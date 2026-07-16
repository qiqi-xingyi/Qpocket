# Author: Yuqi Zhang
"""PredictionPostProcessor — final-stage interpretation, clustering, and export.

Pipeline:
    refinement_result.candidates
        → final validity filter
        → bitstring-level dedup
        → structure-level dedup (CA RMSD)
        → basin clustering (connected components in RMSD graph)
        → per-basin summaries
        → basin ranking by basin_score
        → top-1, top-k candidates, top-k basin representatives
        → optional CA-only PDB export
        → optional CSV / JSON outputs

Does NOT modify sampler / scorer / dense filler / refiner mathematics.
Does NOT reconstruct sidechains / atoms beyond CA.
Does NOT perform global classical optimization.
"""
from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ras_folding.postprocess.clustering import cluster_basins_by_rmsd
from ras_folding.postprocess.dedup import (
    dedup_by_bitstring,
    dedup_by_structure,
)
from ras_folding.postprocess.pdb_export import write_ca_pdb
from ras_folding.postprocess.prediction_types import (
    BasinSummary,
    PredictionResult,
)
from ras_folding.postprocess.selector import rank_basins
from ras_folding.postprocess.summary import compute_basin_summaries
from ras_folding.refinement.refined_types import (
    RefinedCandidate,
    RefinementResult,
)


_EPS_LOG = 1e-12


class PredictionPostProcessor:
    """Final post-processing — interpretation, clustering, selection, export."""

    def __init__(
        self,
        dedup_rmsd_threshold: float = 0.5,
        basin_rmsd_threshold: float = 1.5,
        top_k_candidates: int = 20,
        top_k_basins: int = 5,
        basin_weight_bonus: float = 0.5,
        export_pdb: bool = True,
        ca_atom_name: str = "CA",
    ) -> None:
        if dedup_rmsd_threshold <= 0:
            raise ValueError(
                f"dedup_rmsd_threshold must be > 0; got {dedup_rmsd_threshold}"
            )
        if basin_rmsd_threshold <= 0:
            raise ValueError(
                f"basin_rmsd_threshold must be > 0; got {basin_rmsd_threshold}"
            )
        if top_k_candidates <= 0:
            raise ValueError(
                f"top_k_candidates must be > 0; got {top_k_candidates}"
            )
        if top_k_basins <= 0:
            raise ValueError(
                f"top_k_basins must be > 0; got {top_k_basins}"
            )
        if basin_weight_bonus < 0:
            raise ValueError(
                f"basin_weight_bonus must be >= 0; got {basin_weight_bonus}"
            )
        self.dedup_rmsd_threshold = float(dedup_rmsd_threshold)
        self.basin_rmsd_threshold = float(basin_rmsd_threshold)
        self.top_k_candidates = int(top_k_candidates)
        self.top_k_basins = int(top_k_basins)
        self.basin_weight_bonus = float(basin_weight_bonus)
        self.export_pdb = bool(export_pdb)
        self.ca_atom_name = str(ca_atom_name)

    # ------------------------------------------------------------------ #
    def process(
        self,
        refinement_result: RefinementResult,
        output_dir: Optional[Union[str, Path]] = None,
        sequence: Optional[Union[str, Sequence[str]]] = None,
    ) -> PredictionResult:
        # Deep copy so we never mutate the caller's RefinementResult.
        input_cands: List[RefinedCandidate] = copy.deepcopy(
            list(refinement_result.candidates)
        )
        n_input = len(input_cands)

        # 1. validity filter
        filtered, filter_meta = self._validity_filter(input_cands)

        # 2. bitstring dedup
        bs_dedup = dedup_by_bitstring(filtered)
        n_after_bitstring = len(bs_dedup)

        # 3. structure dedup
        struct_dedup, struct_meta = dedup_by_structure(
            bs_dedup, threshold=self.dedup_rmsd_threshold,
        )
        n_after_structure = len(struct_dedup)

        # 4. basin clustering
        if struct_dedup:
            assignments, cluster_meta = cluster_basins_by_rmsd(
                struct_dedup, threshold=self.basin_rmsd_threshold,
            )
        else:
            assignments = {}
            cluster_meta = {
                "n_basins": 0,
                "basin_rmsd_threshold": self.basin_rmsd_threshold,
                "basin_sizes": [],
                "mean_pairwise_rmsd_global": None,
            }

        # write basin_id back onto candidates for downstream introspection
        for cidx, bid in assignments.items():
            struct_dedup[cidx].basin_id = int(bid)

        # 5. basin summaries + 6. ranking
        unranked = compute_basin_summaries(struct_dedup, assignments)
        ranked = rank_basins(unranked, self.basin_weight_bonus)

        # 7. top-k basin representatives
        basin_reps: List[RefinedCandidate] = []
        for bs in ranked[: self.top_k_basins]:
            basin_reps.append(struct_dedup[bs.representative_index])

        # 8. top-1
        top1: Optional[RefinedCandidate] = (
            struct_dedup[ranked[0].representative_index] if ranked else None
        )

        # 9. top-k candidates by refined_score
        top_candidates = sorted(
            struct_dedup, key=lambda c: c.refined_score,
        )[: self.top_k_candidates]

        # 10. summary
        summary = self._build_summary(
            n_input=n_input,
            filter_meta=filter_meta,
            n_after_bitstring=n_after_bitstring,
            struct_meta=struct_meta,
            cluster_meta=cluster_meta,
            ranked=ranked,
            top1=top1,
            struct_dedup=struct_dedup,
        )

        # 11. outputs
        output_files: Dict[str, str] = {}
        if output_dir is not None:
            output_files = self._write_outputs(
                output_dir=Path(output_dir),
                top1=top1,
                top_candidates=top_candidates,
                basin_reps=basin_reps,
                ranked=ranked,
                struct_dedup=struct_dedup,
                assignments=assignments,
                summary=summary,
                sequence=sequence,
            )

        return PredictionResult(
            top1=top1,
            top_candidates=top_candidates,
            basin_representatives=basin_reps,
            basin_summaries=ranked,
            cluster_assignments=assignments,
            summary=summary,
            output_files=output_files,
        )

    # ================================================================== #
    # internals                                                          #
    # ================================================================== #

    def _validity_filter(
        self, candidates: List[RefinedCandidate],
    ) -> Tuple[List[RefinedCandidate], Dict[str, int]]:
        kept: List[RefinedCandidate] = []
        meta: Dict[str, int] = {
            "n_input_candidates": len(candidates),
            "n_after_validity_filter": 0,
            "n_removed_invalid": 0,
            "n_removed_fallback": 0,
            "n_removed_missing_coords": 0,
            "n_removed_missing_energy": 0,
        }
        for c in candidates:
            s = c.sample
            if s.coords is None:
                meta["n_removed_missing_coords"] += 1
                continue
            if s.full_energy is None:
                meta["n_removed_missing_energy"] += 1
                continue
            if not s.valid or not s.accepted or s.invalid_reason is not None:
                meta["n_removed_invalid"] += 1
                continue
            if c.refined_weight is None or c.refined_weight < 0:
                meta["n_removed_invalid"] += 1
                continue
            if _is_fallback(s):
                meta["n_removed_fallback"] += 1
                continue
            kept.append(c)
        meta["n_after_validity_filter"] = len(kept)
        return kept, meta

    # ------------------------------------------------------------------ #
    def _build_summary(
        self,
        *,
        n_input: int,
        filter_meta: Dict[str, int],
        n_after_bitstring: int,
        struct_meta: Dict[str, Any],
        cluster_meta: Dict[str, Any],
        ranked: List[BasinSummary],
        top1: Optional[RefinedCandidate],
        struct_dedup: List[RefinedCandidate],
    ) -> Dict[str, Any]:
        # refined-weight entropy across struct_dedup
        if struct_dedup:
            w = np.array(
                [float(c.refined_weight) for c in struct_dedup],
                dtype=np.float64,
            )
            w_sum = float(w.sum())
            if w_sum > _EPS_LOG:
                w_n = w / w_sum
                entropy = float(
                    -np.sum(w_n * np.log(w_n + _EPS_LOG))
                )
            else:
                entropy = 0.0
        else:
            entropy = None

        # basin weight diagnostics
        if ranked:
            top_basin = ranked[0]
            top_basin_id = int(top_basin.basin_id)
            top_basin_weight = float(top_basin.basin_weight)
            second_basin_weight = (
                float(ranked[1].basin_weight) if len(ranked) > 1 else None
            )
            if (
                second_basin_weight is not None
                and second_basin_weight > 0
            ):
                top_ratio = top_basin_weight / second_basin_weight
            else:
                top_ratio = None
        else:
            top_basin_id = None
            top_basin_weight = None
            second_basin_weight = None
            top_ratio = None

        # top1 details
        if top1 is not None:
            s = top1.sample
            t1_full = float(s.full_energy)
            t1_score = float(top1.refined_score)
            t1_filter = (
                None if s.filter_energy is None else float(s.filter_energy)
            )
            t1_bs = s.bitstring
            t1_dense = bool(
                (s.metadata.get("densify") or {}).get("is_perturbed", False)
            )
        else:
            t1_full = t1_score = t1_filter = None
            t1_bs = None
            t1_dense = None

        out: Dict[str, Any] = {
            "n_input_candidates": int(n_input),
            "n_after_validity_filter": int(filter_meta["n_after_validity_filter"]),
            "n_removed_invalid": int(filter_meta["n_removed_invalid"]),
            "n_removed_fallback": int(filter_meta["n_removed_fallback"]),
            "n_removed_missing_coords": int(filter_meta["n_removed_missing_coords"]),
            "n_removed_missing_energy": int(filter_meta["n_removed_missing_energy"]),
            "n_after_bitstring_dedup": int(n_after_bitstring),
            "n_after_structure_dedup": int(len(struct_dedup)),
            "structure_dedup_n_merged": int(struct_meta.get("n_merged", 0)),
            "n_basins": int(cluster_meta.get("n_basins", 0)),
            "basin_sizes": list(cluster_meta.get("basin_sizes", [])),
            "mean_pairwise_rmsd_global": cluster_meta.get(
                "mean_pairwise_rmsd_global"
            ),
            "top_basin_id": top_basin_id,
            "top_basin_weight": top_basin_weight,
            "second_basin_weight": second_basin_weight,
            "top_basin_weight_ratio": top_ratio,
            "refined_weight_entropy": entropy,
            "top1_full_energy": t1_full,
            "top1_refined_score": t1_score,
            "top1_filter_energy": t1_filter,
            "top1_bitstring": t1_bs,
            "top1_is_dense": t1_dense,
            "basin_rmsd_threshold": self.basin_rmsd_threshold,
            "dedup_rmsd_threshold": self.dedup_rmsd_threshold,
            "top_k_candidates": self.top_k_candidates,
            "top_k_basins": self.top_k_basins,
            "basin_weight_bonus": self.basin_weight_bonus,
        }
        return out

    # ------------------------------------------------------------------ #
    def _write_outputs(
        self,
        *,
        output_dir: Path,
        top1: Optional[RefinedCandidate],
        top_candidates: List[RefinedCandidate],
        basin_reps: List[RefinedCandidate],
        ranked: List[BasinSummary],
        struct_dedup: List[RefinedCandidate],
        assignments: Dict[int, int],
        summary: Dict[str, Any],
        sequence: Optional[Union[str, Sequence[str]]],
    ) -> Dict[str, str]:
        post_dir = output_dir / "postprocess"
        post_dir.mkdir(parents=True, exist_ok=True)
        files: Dict[str, str] = {}

        # 1. prediction_summary.json
        summary_path = post_dir / "prediction_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(_coerce_json(summary), fh, indent=2)
        files["prediction_summary.json"] = str(summary_path)

        # 2. final_top_candidates.csv
        top_csv = post_dir / "final_top_candidates.csv"
        with top_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([
                "rank", "bitstring", "refined_score", "refined_weight",
                "full_energy", "filter_energy", "basin_id", "is_dense",
                "parent_bitstring", "endpoint_residual", "fallback_triggered",
            ])
            for rank, c in enumerate(top_candidates):
                s = c.sample
                d = s.metadata.get("densify") or {}
                is_dense = bool(d.get("is_perturbed", False))
                parent_bs = d.get("parent_bitstring") if is_dense else None
                w.writerow([
                    rank,
                    s.bitstring or "",
                    f"{c.refined_score:.6f}",
                    f"{c.refined_weight:.6f}",
                    "" if s.full_energy is None else f"{s.full_energy:.6f}",
                    "" if s.filter_energy is None else f"{s.filter_energy:.6f}",
                    "" if c.basin_id is None else int(c.basin_id),
                    "true" if is_dense else "false",
                    parent_bs or "",
                    s.metadata.get("decode_endpoint_residual", ""),
                    "true" if _is_fallback(s) else "false",
                ])
        files["final_top_candidates.csv"] = str(top_csv)

        # 3. basin_summary.csv
        basin_csv = post_dir / "basin_summary.csv"
        with basin_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([
                "basin_rank", "basin_id", "basin_size", "basin_weight",
                "basin_score", "best_refined_score", "best_full_energy",
                "mean_full_energy", "representative_bitstring",
                "mean_pairwise_rmsd",
            ])
            for bs in ranked:
                w.writerow([
                    bs.basin_rank, bs.basin_id, bs.basin_size,
                    f"{bs.basin_weight:.6f}",
                    f"{bs.metadata.get('basin_score', float('nan')):.6f}",
                    f"{bs.best_refined_score:.6f}",
                    f"{bs.best_full_energy:.6f}",
                    f"{bs.mean_full_energy:.6f}",
                    bs.representative_bitstring or "",
                    "" if bs.mean_pairwise_rmsd is None
                    else f"{bs.mean_pairwise_rmsd:.6f}",
                ])
        files["basin_summary.csv"] = str(basin_csv)

        # 4. basin_representatives.csv
        rep_csv = post_dir / "basin_representatives.csv"
        with rep_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([
                "basin_rank", "basin_id", "representative_bitstring",
                "refined_score", "refined_weight", "full_energy",
                "filter_energy", "is_dense", "parent_bitstring",
            ])
            for bs in ranked[: self.top_k_basins]:
                rep = struct_dedup[bs.representative_index]
                d = rep.sample.metadata.get("densify") or {}
                is_dense = bool(d.get("is_perturbed", False))
                w.writerow([
                    bs.basin_rank, bs.basin_id,
                    rep.sample.bitstring or "",
                    f"{rep.refined_score:.6f}",
                    f"{rep.refined_weight:.6f}",
                    "" if rep.sample.full_energy is None
                    else f"{rep.sample.full_energy:.6f}",
                    "" if rep.sample.filter_energy is None
                    else f"{rep.sample.filter_energy:.6f}",
                    "true" if is_dense else "false",
                    d.get("parent_bitstring") if is_dense else "",
                ])
        files["basin_representatives.csv"] = str(rep_csv)

        # 5. cluster_assignments.csv
        ca_csv = post_dir / "cluster_assignments.csv"
        with ca_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([
                "candidate_index", "basin_id", "bitstring",
                "refined_score", "full_energy", "is_dense",
            ])
            for idx, c in enumerate(struct_dedup):
                s = c.sample
                bid = assignments.get(idx, "")
                d = s.metadata.get("densify") or {}
                is_dense = bool(d.get("is_perturbed", False))
                w.writerow([
                    idx, bid, s.bitstring or "",
                    f"{c.refined_score:.6f}",
                    "" if s.full_energy is None
                    else f"{s.full_energy:.6f}",
                    "true" if is_dense else "false",
                ])
        files["cluster_assignments.csv"] = str(ca_csv)

        # 6. CA-only PDBs
        if self.export_pdb:
            if top1 is not None:
                top1_path = post_dir / "top1_ca.pdb"
                write_ca_pdb(
                    top1.sample.coords,
                    sequence,
                    top1_path,
                    atom_name=self.ca_atom_name,
                )
                files["top1_ca.pdb"] = str(top1_path)
            if ranked:
                basins_dir = post_dir / "top_basins"
                basins_dir.mkdir(exist_ok=True)
                for bs in ranked[: self.top_k_basins]:
                    rep = struct_dedup[bs.representative_index]
                    p = basins_dir / f"basin_{bs.basin_rank:03d}_rep_ca.pdb"
                    write_ca_pdb(
                        rep.sample.coords,
                        sequence,
                        p,
                        atom_name=self.ca_atom_name,
                    )
                    files[f"top_basins/basin_{bs.basin_rank:03d}_rep_ca.pdb"] = (
                        str(p)
                    )

        return files


# ---------------------------------------------------------------------- #
# helpers                                                                #
# ---------------------------------------------------------------------- #

def _is_fallback(sample) -> bool:
    """Detect fallback decode either from flat metadata keys
    (decode_fallback_triggered) or nested decode_info.fallback_triggered."""
    if sample.metadata.get("decode_fallback_triggered") is True:
        return True
    if sample.metadata.get("fallback_triggered") is True:
        return True
    info = sample.metadata.get("decode_info")
    if isinstance(info, dict) and info.get("fallback_triggered") is True:
        return True
    return False


def _coerce_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _coerce_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_json(x) for x in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


__all__ = ["PredictionPostProcessor"]
