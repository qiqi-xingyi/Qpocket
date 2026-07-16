# Author: Yuqi Zhang
"""PerturbationDenseFiller — generate child candidates around top quantum parents.

Pipeline (single call to ``densify``):

    eligible parents
        = candidates with valid=True, accepted=True, coords≠None, full_energy≠None
    sort parents by refined_score (if present in metadata) ASC, else by full_energy ASC
    keep top_parents

    for each parent:
        for k in range(children_per_parent):
            sigma_k = rng.choice(angular_sigmas_deg)
            child_coords = perturb_bond_vectors(parent.coords, sigma_k, rng)
            run validate_decoded_coords on child_coords; reject if invalid
            local_rmsd = local_ca_rmsd(parent.coords, child_coords)
            reject if local_rmsd > max_local_rmsd
            create child CandidateSample with valid=True, accepted=True
            run scorer.evaluate(child)
            energy_delta = child.full_energy - parent.full_energy
            reject if energy_delta > energy_window
            keep child

    apply weight policy (per-parent mass conservation)
    return DenseFillResult

This module never mutates the SCIENTIFIC behavior of FilterHamiltonian,
FullEnergyScorer, or SubspaceDiagonalizationRefiner. Dense children flow
through the same scorer used for parents and through the same refiner.
The filler is OPTIONAL — runners can skip it entirely.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ras_folding.densify.angular_perturb import (
    local_ca_rmsd,
    perturb_bond_vectors,
)
from ras_folding.densify.dense_types import DenseFillResult
from ras_folding.densify.weight_policy import assign_dense_weights
from ras_folding.sampler.context import get_encoder_inputs
from ras_folding.sampler.sample_types import CandidateSample
from ras_folding.sampler.validity import validate_decoded_coords


# Sentinel marker on dense-child bitstrings so they don't collide with
# real quantum bitstrings during dedup. The character "#" is invalid in
# binary bitstrings, so any bitstring_str_to_int call on these will fail
# loudly — preventing accidental decoding of a perturbed child.
_DENSE_CHILD_TAG: str = "#d"


class PerturbationDenseFiller:
    """Optional dense-fill stage between full scoring and SQD refinement."""

    def __init__(
        self,
        top_parents: int = 100,
        children_per_parent: int = 20,
        angular_sigmas_deg: Tuple[float, ...] = (2.0, 5.0, 10.0),
        max_local_rmsd: float = 1.0,
        endpoint_tolerance: float = 1.0,
        energy_window: float = 10.0,
        perturbation_mass: float = 0.3,
        seed: Optional[int] = None,
    ) -> None:
        if top_parents <= 0:
            raise ValueError(
                f"top_parents must be > 0; got {top_parents}"
            )
        if children_per_parent <= 0:
            raise ValueError(
                f"children_per_parent must be > 0; got {children_per_parent}"
            )
        if not angular_sigmas_deg:
            raise ValueError("angular_sigmas_deg must be non-empty")
        for s in angular_sigmas_deg:
            if s < 0:
                raise ValueError(
                    f"angular_sigmas_deg must be >= 0; got {s}"
                )
        if max_local_rmsd <= 0:
            raise ValueError(
                f"max_local_rmsd must be > 0; got {max_local_rmsd}"
            )
        if energy_window <= 0:
            raise ValueError(
                f"energy_window must be > 0; got {energy_window}"
            )
        if not (0.0 <= perturbation_mass <= 1.0):
            raise ValueError(
                f"perturbation_mass must be in [0, 1]; got {perturbation_mass}"
            )

        self.top_parents = int(top_parents)
        self.children_per_parent = int(children_per_parent)
        self.angular_sigmas_deg = tuple(float(s) for s in angular_sigmas_deg)
        self.max_local_rmsd = float(max_local_rmsd)
        self.endpoint_tolerance = float(endpoint_tolerance)
        self.energy_window = float(energy_window)
        self.perturbation_mass = float(perturbation_mass)
        self.seed = seed

    # ------------------------------------------------------------------ #
    def densify(
        self,
        candidates: Sequence[CandidateSample],
        scorer,
        context_or_inputs,
    ) -> DenseFillResult:
        # Ensure encoder inputs are resolvable — surfaces a clear error
        # before we burn time perturbing.
        _ = get_encoder_inputs(context_or_inputs)

        n_input = len(candidates)
        eligible = [
            s for s in candidates
            if s.valid
            and s.accepted
            and s.coords is not None
            and s.full_energy is not None
        ]

        # Sort by refined_score if available, else by full_energy ASC.
        def _key(s: CandidateSample):
            rs = s.metadata.get("refined_score")
            if isinstance(rs, (int, float)):
                return (0, float(rs), float(s.full_energy))
            return (1, float(s.full_energy), 0.0)

        eligible.sort(key=_key)
        parents = eligible[: self.top_parents]

        rng = np.random.default_rng(self.seed)

        children: List[CandidateSample] = []
        n_children_generated = 0
        n_children_valid = 0
        rejection_counts: Dict[str, int] = {}
        local_rmsds: List[float] = []
        energy_deltas: List[float] = []

        sigmas = np.asarray(self.angular_sigmas_deg, dtype=np.float64)

        for parent_rank, parent in enumerate(parents):
            for child_idx in range(self.children_per_parent):
                n_children_generated += 1
                sigma = float(rng.choice(sigmas))
                try:
                    child_coords = perturb_bond_vectors(
                        np.asarray(parent.coords, dtype=np.float64),
                        sigma_deg=sigma,
                        rng=rng,
                    )
                except ValueError as e:
                    rejection_counts["perturb_failed"] = (
                        rejection_counts.get("perturb_failed", 0) + 1
                    )
                    continue

                valid, reason, _info = validate_decoded_coords(
                    child_coords, context_or_inputs,
                )
                if not valid:
                    key = reason or "unknown_invalid"
                    rejection_counts[key] = (
                        rejection_counts.get(key, 0) + 1
                    )
                    continue
                n_children_valid += 1

                rmsd = local_ca_rmsd(parent.coords, child_coords)
                if rmsd > self.max_local_rmsd:
                    rejection_counts["max_local_rmsd"] = (
                        rejection_counts.get("max_local_rmsd", 0) + 1
                    )
                    continue

                child = self._build_child(
                    parent=parent,
                    parent_rank=parent_rank,
                    child_index=child_idx,
                    child_coords=child_coords,
                    sigma_deg=sigma,
                    local_rmsd=rmsd,
                )
                # rescore via the same FullEnergyScorer used for parents
                try:
                    e_full, full_terms = scorer.evaluate(
                        child, context_or_inputs,
                    )
                except Exception as e:
                    rejection_counts["scorer_exception"] = (
                        rejection_counts.get("scorer_exception", 0) + 1
                    )
                    child.metadata["scorer_exception"] = repr(e)
                    continue
                child.full_energy = float(e_full)
                child.full_energy_terms = dict(full_terms)

                energy_delta = float(e_full) - float(parent.full_energy)
                child.metadata["densify"]["energy_delta"] = energy_delta
                if energy_delta > self.energy_window:
                    rejection_counts["energy_window"] = (
                        rejection_counts.get("energy_window", 0) + 1
                    )
                    continue

                local_rmsds.append(rmsd)
                energy_deltas.append(energy_delta)
                children.append(child)

        # Apply weight policy on accepted children + their parents.
        assign_dense_weights(
            parents, children, perturbation_mass=self.perturbation_mass,
        )

        # Mark parents so refiner diagnostics can count them.
        for p in parents:
            p.metadata.setdefault("densify_parent_marker", True)

        all_candidates: List[CandidateSample] = list(parents) + list(children)

        summary: Dict[str, Any] = {
            "n_parent_input": int(n_input),
            "n_parent_eligible": int(len(eligible)),
            "n_parent_selected": int(len(parents)),
            "n_children_generated": int(n_children_generated),
            "n_children_valid": int(n_children_valid),
            "n_children_kept": int(len(children)),
            "n_children_rejected": int(n_children_generated - len(children)),
            "rejection_reason_counts": rejection_counts,
            "mean_local_rmsd_to_parent": (
                float(np.mean(local_rmsds)) if local_rmsds else None
            ),
            "max_local_rmsd_to_parent": (
                float(np.max(local_rmsds)) if local_rmsds else None
            ),
            "mean_energy_delta": (
                float(np.mean(energy_deltas)) if energy_deltas else None
            ),
            "best_energy_delta": (
                float(np.min(energy_deltas)) if energy_deltas else None
            ),
            "perturbation_mass": float(self.perturbation_mass),
            "angular_sigmas_deg": list(self.angular_sigmas_deg),
            "max_local_rmsd": float(self.max_local_rmsd),
            "energy_window": float(self.energy_window),
            "top_parents": int(self.top_parents),
            "children_per_parent": int(self.children_per_parent),
        }

        return DenseFillResult(
            parent_candidates=parents,
            dense_candidates=children,
            all_candidates=all_candidates,
            summary=summary,
        )

    # ------------------------------------------------------------------ #
    def _build_child(
        self,
        *,
        parent: CandidateSample,
        parent_rank: int,
        child_index: int,
        child_coords: np.ndarray,
        sigma_deg: float,
        local_rmsd: float,
    ) -> CandidateSample:
        """Construct a CandidateSample for a perturbed child.

        - bitstring uses a `#d{idx}` suffix on the parent bitstring so
          dedup in the refiner does NOT collapse children into parent.
        - codes are inherited (a fresh list copy).
        - filter_energy is None: dense children do NOT participate in the
          H_filter/τ acceptance loop. metadata.densify.filter_excluded
          surfaces this fact.
        """
        parent_bs = parent.bitstring
        if parent_bs is None:
            parent_bs = f"id{id(parent)}"
        child_bs = f"{parent_bs}{_DENSE_CHILD_TAG}{child_index}"
        child_codes = list(parent.codes) if parent.codes is not None else None

        meta: Dict[str, Any] = {
            "densify": {
                "is_perturbed": True,
                "parent_bitstring": parent.bitstring,
                "parent_full_energy": float(parent.full_energy),
                "parent_filter_energy": (
                    None if parent.filter_energy is None
                    else float(parent.filter_energy)
                ),
                "parent_refined_weight": parent.metadata.get(
                    "refined_weight",
                ),
                "perturbation_type": "angular",
                "angular_sigma_deg": float(sigma_deg),
                "local_rmsd_to_parent": float(local_rmsd),
                "child_index": int(child_index),
                "parent_rank": int(parent_rank),
                "filter_excluded": True,
            }
        }
        return CandidateSample(
            bitstring=child_bs,
            codes=child_codes,
            coords=child_coords,
            count=1,
            base_probability=None,
            filtered_probability=None,
            accepted=True,
            valid=True,
            invalid_reason=None,
            filter_energy=None,
            full_energy=None,             # filled by scorer.evaluate above
            filter_terms={},
            full_energy_terms={},
            metadata=meta,
        )


# ---------------------------------------------------------------------- #
# output helpers                                                         #
# ---------------------------------------------------------------------- #

def write_dense_outputs(
    result: DenseFillResult,
    output_dir: Path,
) -> Dict[str, str]:
    """Persist DenseFillResult to disk.

    Writes
    ------
    densify_summary.json
    dense_candidates.csv  (per-child row; never mixes parents with children)

    The CSV intentionally omits parent rows — parents already exist in
    the upstream sample log. The CSV rationale is to give a side-by-side
    parent/child comparison for the SAME parent_bitstring.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "densify_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(_summary_for_json(result.summary), fh, indent=2)

    csv_path = output_dir / "dense_candidates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "parent_bitstring",
            "child_index",
            "perturbation_type",
            "angular_sigma_deg",
            "local_rmsd_to_parent",
            "parent_full_energy",
            "child_full_energy",
            "energy_delta",
            "valid",
            "invalid_reason",
            "dense_weight",
            "parent_rank",
        ])
        for c in result.dense_candidates:
            d = c.metadata.get("densify", {})
            w.writerow([
                d.get("parent_bitstring", ""),
                d.get("child_index", ""),
                d.get("perturbation_type", ""),
                d.get("angular_sigma_deg", ""),
                d.get("local_rmsd_to_parent", ""),
                d.get("parent_full_energy", ""),
                "" if c.full_energy is None else f"{c.full_energy:.6f}",
                d.get("energy_delta", ""),
                "true" if c.valid else "false",
                c.invalid_reason or "",
                d.get("dense_weight", ""),
                d.get("parent_rank", ""),
            ])

    files = {
        "densify_summary.json": str(summary_path),
        "dense_candidates.csv": str(csv_path),
    }
    result.output_files.update(files)
    return files


def _summary_for_json(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort JSON serialisation. Casts numpy scalars to floats."""
    out: Dict[str, Any] = {}
    for k, v in summary.items():
        if isinstance(v, dict):
            out[k] = {str(kk): _coerce_for_json(vv) for kk, vv in v.items()}
        else:
            out[k] = _coerce_for_json(v)
    return out


def _coerce_for_json(v: Any) -> Any:
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    return v


__all__ = ["PerturbationDenseFiller", "write_dense_outputs"]
