# Author: Yuqi Zhang
"""oracle_eval dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class OracleCandidate:
    task_id: str
    source: str   # "accepted" | "dense" | "refined" | "postprocess"
    bitstring: Optional[str]
    coords: np.ndarray
    full_energy: Optional[float] = None
    refined_score: Optional[float] = None
    refined_weight: Optional[float] = None
    is_dense: bool = False
    parent_bitstring: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OracleBestResult:
    task_id: str
    best_candidate: Optional[OracleCandidate]
    best_rmsd: Optional[float]
    reference_available: bool
    n_candidates_considered: int
    output_pdb: Optional[Path] = None
    summary: Dict[str, Any] = field(default_factory=dict)


__all__ = ["OracleCandidate", "OracleBestResult"]
