# Author: Yuqi Zhang
"""Result dataclasses for subspace refinement."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import scipy.sparse as sp

from ras_folding.sampler.sample_types import CandidateSample


@dataclass
class RefinedCandidate:
    """One candidate after subspace diagonalization.

    Attributes
    ----------
    sample : the underlying CandidateSample (must satisfy
             CandidateSample.is_eligible_for_refinement()).
    refined_weight : sum_k mode_boltzmann_weight_k * |v_k[i]|^2 (normalized
                     to sum=1 across the subspace).
    refined_score : E_norm_i - alpha * log(refined_weight + eps)
    mode_weights : (n_modes,) per-mode contribution to refined_weight,
                   i.e. mode_boltzmann_weight_k * |v_k[i]|^2 (NOT
                   normalized — the row sum is the un-normalized
                   refined_weight).
    basin_id : optional integer cluster id (for downstream basin grouping).
    metadata : free-form provenance.
    """
    sample: CandidateSample
    refined_weight: float
    refined_score: float
    mode_weights: np.ndarray
    basin_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RefinementResult:
    """Result of SubspaceDiagonalizationRefiner.refine().

    Fields
    ------
    candidates : list of RefinedCandidate, sorted by ascending refined_score.
    eigenvalues : (n_modes,) float, ascending.
    eigenvectors : (N, n_modes) float, columns are modes.
    effective_hamiltonian : (N, N) sparse OR dense matrix.
    selected_indices : indices into the input `eligible_samples` order
        showing which samples ended up in the subspace S.
    parameters : runtime parameters (kappa, sigma, k_neighbors, ...).
    summary : statistics (n_eligible, n_selected, n_nonzero_couplings, ...).
    """
    candidates: List[RefinedCandidate]
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    effective_hamiltonian: Union[np.ndarray, sp.spmatrix]
    selected_indices: List[int]
    parameters: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)


__all__ = ["RefinedCandidate", "RefinementResult"]
