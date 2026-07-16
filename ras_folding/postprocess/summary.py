# Author: Yuqi Zhang
"""Per-basin summaries for the post-processed prediction."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from ras_folding.postprocess.prediction_types import BasinSummary
from ras_folding.postprocess.rmsd import pairwise_ca_rmsd
from ras_folding.refinement.refined_types import RefinedCandidate


def compute_basin_summaries(
    candidates: Sequence[RefinedCandidate],
    cluster_assignments: Dict[int, int],
) -> List[BasinSummary]:
    """Aggregate basin-level statistics.

    Parameters
    ----------
    candidates : indexable list of RefinedCandidate (post-dedup).
    cluster_assignments : dict[candidate_index → basin_id].

    Returns
    -------
    list[BasinSummary] indexed by basin_id (ascending), with
    ``basin_rank`` initialised to basin_id (i.e. unranked) — the caller
    runs ``rank_basins`` to set the final ordering.

    Representative selection
    ------------------------
    Within a basin, the representative is the candidate with the lowest
    refined_score; ties broken by lowest full_energy.
    """
    if not candidates:
        return []

    members_by_basin: Dict[int, List[int]] = {}
    for cidx, bid in cluster_assignments.items():
        members_by_basin.setdefault(bid, []).append(cidx)

    out: List[BasinSummary] = []
    for bid in sorted(members_by_basin.keys()):
        members = members_by_basin[bid]
        # representative: lowest refined_score, tie-break lowest full_energy
        rep_idx = min(
            members,
            key=lambda i: (
                candidates[i].refined_score,
                candidates[i].sample.full_energy,
            ),
        )
        rep = candidates[rep_idx]
        weights = np.array(
            [float(candidates[i].refined_weight) for i in members],
            dtype=np.float64,
        )
        full_energies = np.array(
            [float(candidates[i].sample.full_energy) for i in members],
            dtype=np.float64,
        )
        refined_scores = np.array(
            [float(candidates[i].refined_score) for i in members],
            dtype=np.float64,
        )

        if len(members) >= 2:
            sub = [candidates[i] for i in members]
            rmsd = pairwise_ca_rmsd(sub)
            iu, ju = np.triu_indices(len(members), k=1)
            mean_rmsd = (
                float(rmsd[iu, ju].mean()) if iu.size else 0.0
            )
        else:
            mean_rmsd = 0.0

        out.append(BasinSummary(
            basin_id=int(bid),
            basin_rank=int(bid),
            basin_size=int(len(members)),
            basin_weight=float(weights.sum()),
            best_refined_score=float(refined_scores.min()),
            best_full_energy=float(full_energies.min()),
            mean_full_energy=float(full_energies.mean()),
            representative_index=int(rep_idx),
            representative_bitstring=rep.sample.bitstring,
            mean_pairwise_rmsd=mean_rmsd,
            metadata={"member_indices": [int(i) for i in members]},
        ))

    return out


__all__ = ["compute_basin_summaries"]
