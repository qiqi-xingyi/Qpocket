# Author: Yuqi Zhang
"""Result dataclasses for the prediction post-processor."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ras_folding.refinement.refined_types import RefinedCandidate


@dataclass
class BasinSummary:
    basin_id: int
    basin_rank: int
    basin_size: int
    basin_weight: float
    best_refined_score: float
    best_full_energy: float
    mean_full_energy: float
    representative_index: int
    representative_bitstring: Optional[str]
    mean_pairwise_rmsd: Optional[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    top1: Optional[RefinedCandidate]
    top_candidates: List[RefinedCandidate]
    basin_representatives: List[RefinedCandidate]
    basin_summaries: List[BasinSummary]
    cluster_assignments: Dict[int, int]
    summary: Dict[str, Any]
    output_files: Dict[str, str] = field(default_factory=dict)


__all__ = ["BasinSummary", "PredictionResult"]
