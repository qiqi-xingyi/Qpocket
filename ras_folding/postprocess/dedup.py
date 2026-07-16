# Author: Yuqi Zhang
"""Bitstring-level and structure-level dedup of RefinedCandidates.

Both functions return NEW lists; they mutate ``metadata`` of the
representative RefinedCandidate to record dedup provenance, but they do
NOT mutate the input order or merge weights into discarded duplicates.
The caller is expected to deep-copy candidates upstream if they want
isolation from the original RefinementResult — see PredictionPostProcessor.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ras_folding.postprocess.rmsd import ca_rmsd
from ras_folding.refinement.refined_types import RefinedCandidate


def dedup_by_bitstring(
    candidates: Sequence[RefinedCandidate],
) -> List[RefinedCandidate]:
    """Collapse duplicates by sample.bitstring.

    Rules:
      - bitstring=None ⇒ NEVER dedup (keep all such candidates).
      - representative within a duplicate group = lowest refined_score.
      - representative.refined_weight += sum(duplicates.refined_weight).
      - representative.metadata["dedup"] records:
            merged_bitstring_count, merged_from_indices.
      - Original input list NOT mutated; returned representatives are the
        same RefinedCandidate objects as input (mutated for metadata).
    """
    if not candidates:
        return []

    # Group by bitstring; None bitstrings each form their own singleton group.
    groups: Dict[Any, List[Tuple[int, RefinedCandidate]]] = {}
    none_singletons: List[Tuple[int, RefinedCandidate]] = []
    for idx, c in enumerate(candidates):
        bs = c.sample.bitstring
        if bs is None:
            none_singletons.append((idx, c))
            continue
        groups.setdefault(bs, []).append((idx, c))

    out: List[RefinedCandidate] = []
    # First handle the bitstring groups (preserve the order in which a
    # bitstring is FIRST encountered, so output order is stable).
    seen_bs_order: List[Any] = []
    for idx, c in enumerate(candidates):
        bs = c.sample.bitstring
        if bs is None or bs in seen_bs_order:
            continue
        seen_bs_order.append(bs)

    for bs in seen_bs_order:
        members = groups[bs]
        members_sorted = sorted(members, key=lambda p: p[1].refined_score)
        rep_idx, rep = members_sorted[0]
        merged_count = len(members) - 1
        if merged_count > 0:
            merged_weight = sum(p[1].refined_weight for p in members_sorted[1:])
            rep.refined_weight = float(rep.refined_weight + merged_weight)
            dedup_meta = rep.metadata.setdefault("dedup", {})
            dedup_meta["merged_bitstring_count"] = int(merged_count)
            dedup_meta["merged_from_indices"] = [
                int(p[0]) for p in members_sorted[1:]
            ]
        out.append(rep)

    # Then append all None-bitstring singletons in original order.
    for _idx, c in none_singletons:
        out.append(c)

    return out


def dedup_by_structure(
    candidates: Sequence[RefinedCandidate],
    threshold: float,
) -> Tuple[List[RefinedCandidate], Dict[str, Any]]:
    """Greedy CA-RMSD dedup. Keep best refined_score, merge weights.

    Algorithm:
      1. Sort candidates by refined_score ASC (best first).
      2. Iterate; for each candidate, compute RMSD against every
         already-kept representative.
      3. If min RMSD < threshold ⇒ merge into nearest representative
         (weight accumulates, metadata records the merge).
      4. Else ⇒ keep as a new representative.

    Notes
    -----
    The output preserves the BEST-FIRST order (by refined_score), which
    is what the basin-clustering step expects. Threshold is strict-less-than.
    """
    if not candidates:
        return [], {
            "n_before_structure_dedup": 0,
            "n_after_structure_dedup": 0,
            "dedup_rmsd_threshold": float(threshold),
            "n_merged": 0,
        }

    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda p: p[1].refined_score)

    reps: List[RefinedCandidate] = []
    rep_orig_idx: List[int] = []
    n_merged = 0

    for orig_idx, c in indexed:
        if not reps:
            reps.append(c)
            rep_orig_idx.append(orig_idx)
            continue
        best_d = float("inf")
        best_j = -1
        for j, r in enumerate(reps):
            d = ca_rmsd(c.sample.coords, r.sample.coords)
            if d < best_d:
                best_d = d
                best_j = j
        if best_d < threshold:
            # merge into rep[best_j]
            rep = reps[best_j]
            rep.refined_weight = float(rep.refined_weight + c.refined_weight)
            dedup_meta = rep.metadata.setdefault("dedup", {})
            dedup_meta.setdefault("merged_structure_count", 0)
            dedup_meta["merged_structure_count"] += 1
            dedup_meta.setdefault("merged_structure_indices", []).append(
                int(orig_idx)
            )
            n_merged += 1
        else:
            reps.append(c)
            rep_orig_idx.append(orig_idx)

    summary: Dict[str, Any] = {
        "n_before_structure_dedup": len(candidates),
        "n_after_structure_dedup": len(reps),
        "dedup_rmsd_threshold": float(threshold),
        "n_merged": int(n_merged),
    }
    return reps, summary


__all__ = ["dedup_by_bitstring", "dedup_by_structure"]
