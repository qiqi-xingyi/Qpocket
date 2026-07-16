# Author: Yuqi Zhang
"""StructureAnalyzer — per-task structural diagnostics on the post-processed
prediction (top-1, top-5, basins, geometry sanity)."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ras_folding.postprocess.prediction_types import PredictionResult
from ras_folding.postprocess.rmsd import ca_rmsd, pairwise_ca_rmsd
from ras_folding.utils.constants import CA_CA_LENGTH


_EPS = 1e-12


def _is_dense(sample) -> bool:
    return bool(
        (sample.metadata.get("densify") or {}).get("is_perturbed", False)
    )


def _bond_lengths(coords: np.ndarray) -> np.ndarray:
    return np.linalg.norm(coords[1:] - coords[:-1], axis=1)


def _min_nonlocal_distance(coords: np.ndarray, gap: int = 3) -> float:
    n = coords.shape[0]
    if n < gap + 1:
        return float("inf")
    iu, ju = np.triu_indices(n, k=gap)
    if iu.size == 0:
        return float("inf")
    d = np.linalg.norm(coords[iu] - coords[ju], axis=1)
    return float(d.min())


def _radius_of_gyration(coords: np.ndarray) -> float:
    if coords.shape[0] == 0:
        return 0.0
    center = coords.mean(axis=0)
    return float(math.sqrt(float(np.mean(np.sum((coords - center) ** 2, axis=1)))))


class StructureAnalyzer:
    def __init__(self, *, long_range_min_sep: float = 3.8) -> None:
        self.long_range_min_sep = float(long_range_min_sep)

    # ------------------------------------------------------------------ #
    def analyze(
        self,
        prediction_result: PredictionResult,
        reference_coords: Optional[np.ndarray],
        output_dir: Path,
    ) -> Dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out: Dict[str, Any] = {}

        top1 = prediction_result.top1
        candidates = prediction_result.top_candidates
        basin_summaries = prediction_result.basin_summaries

        # ------ 1. top-1 metrics ---------------------------------------
        if top1 is not None:
            s = top1.sample
            coords = np.asarray(s.coords, dtype=np.float64)
            t1: Dict[str, Any] = {
                "bitstring": s.bitstring,
                "full_energy": _opt_float(s.full_energy),
                "refined_score": float(top1.refined_score),
                "refined_weight": float(top1.refined_weight),
                "is_dense": _is_dense(s),
                "parent_bitstring": (
                    (s.metadata.get("densify") or {}).get("parent_bitstring")
                ),
                "endpoint_residual": s.metadata.get("decode_endpoint_residual"),
                "fallback_triggered": bool(
                    s.metadata.get("decode_fallback_triggered")
                    or s.metadata.get("fallback_triggered")
                ),
                "rg": _radius_of_gyration(coords),
                "filter_energy": _opt_float(s.filter_energy),
            }
            full_terms = s.full_energy_terms or {}
            for k in ("contact_full", "overlap_full"):
                if k in full_terms:
                    t1[k] = float(full_terms[k])
            filter_terms = s.filter_terms or {}
            if "favorable_contact_miss" in filter_terms:
                t1["favorable_contact_miss"] = float(
                    filter_terms["favorable_contact_miss"]
                )
            if reference_coords is not None and reference_coords.shape == coords.shape:
                t1["rmsd_to_reference"] = ca_rmsd(coords, reference_coords)
            out["top1"] = t1
        else:
            out["top1"] = None

        # ------ 2. top-k metrics ---------------------------------------
        if candidates:
            k = min(5, len(candidates))
            top_k = candidates[:k]
            coords_stack = [
                np.asarray(c.sample.coords, dtype=np.float64) for c in top_k
            ]
            full_es = [
                float(c.sample.full_energy)
                for c in top_k if c.sample.full_energy is not None
            ]
            n_dense = sum(_is_dense(c.sample) for c in top_k)
            tk_meta: Dict[str, Any] = {
                "k": int(k),
                "best_full_energy": (
                    min(full_es) if full_es else None
                ),
                "dense_fraction": float(n_dense) / float(k),
            }
            if k >= 2:
                R = pairwise_ca_rmsd(top_k)
                iu, ju = np.triu_indices(k, k=1)
                tk_meta["mean_pairwise_rmsd"] = float(R[iu, ju].mean())
            else:
                tk_meta["mean_pairwise_rmsd"] = 0.0
            if reference_coords is not None:
                rms = []
                for c in top_k:
                    coords = np.asarray(c.sample.coords, dtype=np.float64)
                    if coords.shape == reference_coords.shape:
                        rms.append(ca_rmsd(coords, reference_coords))
                tk_meta["best_rmsd"] = float(min(rms)) if rms else None
                tk_meta["mean_rmsd"] = float(np.mean(rms)) if rms else None
            out["top_k"] = tk_meta
        else:
            out["top_k"] = None

        # ------ 3. basin metrics ---------------------------------------
        if basin_summaries:
            n = len(basin_summaries)
            weights = np.array(
                [bs.basin_weight for bs in basin_summaries], dtype=np.float64,
            )
            w_sum = float(weights.sum())
            if w_sum > _EPS:
                w_n = weights / w_sum
                entropy = float(-np.sum(w_n * np.log(w_n + _EPS)))
            else:
                entropy = 0.0
            top_w = float(weights[0]) if n >= 1 else 0.0
            second_w = float(weights[1]) if n >= 2 else 0.0
            ratio = (top_w / second_w) if second_w > _EPS else None
            basin_meta: Dict[str, Any] = {
                "n_basins": int(n),
                "top_basin_weight": top_w,
                "second_basin_weight": second_w,
                "basin_weight_ratio": ratio,
                "basin_entropy": entropy,
            }
            if reference_coords is not None:
                rmsds: List[Any] = []
                for bs in basin_summaries:
                    rep_idx = int(bs.representative_index)
                    rep = candidates[rep_idx] if rep_idx < len(candidates) else None
                    if rep is None or rep.sample.coords is None:
                        rmsds.append(None)
                        continue
                    coords = np.asarray(rep.sample.coords, dtype=np.float64)
                    if coords.shape == reference_coords.shape:
                        rmsds.append(ca_rmsd(coords, reference_coords))
                    else:
                        rmsds.append(None)
                basin_meta["basin_representative_rmsds"] = rmsds
            out["basins"] = basin_meta
        else:
            out["basins"] = None

        # ------ 4. geometry sanity (top-1) -----------------------------
        if top1 is not None:
            coords = np.asarray(top1.sample.coords, dtype=np.float64)
            bl = _bond_lengths(coords)
            min_nl = _min_nonlocal_distance(coords, gap=3)
            geom: Dict[str, Any] = {
                "mean_bond_length_top1": (
                    float(bl.mean()) if bl.size else None
                ),
                "max_bond_length_error_top1": (
                    float(np.max(np.abs(bl - CA_CA_LENGTH)))
                    if bl.size else None
                ),
                "min_nonlocal_distance_top1": (
                    None if math.isinf(min_nl) else float(min_nl)
                ),
                "long_range_clash_top1": (
                    bool(min_nl < self.long_range_min_sep)
                    if not math.isinf(min_nl) else False
                ),
            }
            out["geometry_sanity"] = geom
        else:
            out["geometry_sanity"] = None

        # ------ persist -----------------------------------------------
        path = output_dir / "structure_analysis.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(_coerce_json(out), fh, indent=2)
        out["__output_path__"] = str(path)
        return out


def _opt_float(x):
    return None if x is None else float(x)


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
    return o


__all__ = ["StructureAnalyzer"]
