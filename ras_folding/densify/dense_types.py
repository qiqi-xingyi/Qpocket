# Author: Yuqi Zhang
"""Dense-fill result dataclass."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from ras_folding.sampler.sample_types import CandidateSample


@dataclass
class DenseFillResult:
    """Output of PerturbationDenseFiller.densify.

    Fields
    ------
    parent_candidates : list[CandidateSample]
        The post-filter parents that were actually used to spawn children.
        These are the SAME objects supplied as input (after metadata
        annotation by the weight policy) — not deep copies.
    dense_candidates : list[CandidateSample]
        The newly generated child candidates that survived validity,
        local-RMSD, and energy-window filtering.
    all_candidates : list[CandidateSample]
        parent_candidates + dense_candidates, in that order. The downstream
        SubspaceDiagonalizationRefiner can refine on this list directly.
    summary : dict[str, Any]
        Headline statistics — see PerturbationDenseFiller.densify docstring
        for the full key list.
    output_files : dict[str, str]
        Paths of any files written by write_dense_outputs.
    """
    parent_candidates: List[CandidateSample]
    dense_candidates: List[CandidateSample]
    all_candidates: List[CandidateSample]
    summary: Dict[str, Any] = field(default_factory=dict)
    output_files: Dict[str, str] = field(default_factory=dict)


__all__ = ["DenseFillResult"]
