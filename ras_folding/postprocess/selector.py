# Author: Yuqi Zhang
"""Basin ranking by basin_score = best_refined_score - bonus * log(weight)."""
from __future__ import annotations

import math
from typing import List, Sequence

from ras_folding.postprocess.prediction_types import BasinSummary


_EPS_LOG = 1e-12


def rank_basins(
    basin_summaries: Sequence[BasinSummary],
    basin_weight_bonus: float = 0.5,
) -> List[BasinSummary]:
    """Compute basin_score for every summary and return them sorted ASC.

        basin_score_b = best_refined_score_b - bonus * log(basin_weight_b + eps)

    Side effect: each summary's metadata["basin_score"] is set, and
    basin_rank is overwritten with the position in the sorted list.
    Returns the (same) summaries, re-ordered.
    """
    if basin_weight_bonus < 0:
        raise ValueError(
            f"basin_weight_bonus must be >= 0; got {basin_weight_bonus}"
        )
    summaries = list(basin_summaries)
    for s in summaries:
        w = float(s.basin_weight)
        # Guard against negative / zero weights — should not happen with
        # post-refinement weights, but defensive.
        score = float(s.best_refined_score) - float(basin_weight_bonus) * math.log(
            max(w, 0.0) + _EPS_LOG
        )
        s.metadata["basin_score"] = float(score)

    summaries.sort(key=lambda s: s.metadata["basin_score"])
    for rank, s in enumerate(summaries):
        s.basin_rank = int(rank)
    return summaries


__all__ = ["rank_basins"]
