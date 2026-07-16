# Author: Yuqi Zhang
"""Lightweight reachability + clash filter for autoregressive walk.

Replaces the strict (vox, dir_bin) DP. At this stage we only rule out
candidates that are absolutely impossible; everything else is left to
the quantum sampler under H_geom soft penalties.

What this module enforces (HARD, before bit selection):
  - bend angle: implicit via lattice annulus [BEND_MIN_DEG, BEND_MAX_DEG]
    (every lattice candidate is bend-legal by construction)
  - reach to anchor_right: candidate must be able to extend in the
    remaining bonds, with bond length in [L_MIN, L_MAX]
  - long-range self-clash: candidate must be at least MIN_SEP from any
    non-recent prior CA (skip last SKIP_RECENT history entries — they
    are bend-protected and cannot clash by construction)

What it does NOT enforce (left to H_geom):
  - torsion / dihedral preferences
  - radius of gyration
  - clashes with environment atoms (ligand, other chains)
  - i,i+1 / i,i+2 distance precision (already pinned by L=CA_CA_LENGTH
    and bend window)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

import math

from ras_folding.encoder.inputs import EncoderInputs
from ras_folding.encoder.lattice import (
    BEND_MAX_DEG, BEND_MIN_DEG, N_DIRECTIONS, lattice_around,
)
from ras_folding.utils.constants import CA_CA_LENGTH

_COS_BEND_MIN = math.cos(math.radians(BEND_MIN_DEG))
_COS_BEND_MAX = math.cos(math.radians(BEND_MAX_DEG))


# --- per-step constraint constants --------------------------------------
# Bond length window — measured across 35 native KRAS fragments
# (n=357 bonds): [3.735, 3.898] A. The window is generous on both sides
# so reach checks don't accidentally reject near-native candidates.
L_MIN: float = 3.70
L_MAX: float = 3.90

# Long-range self-clash threshold. Native gap>=3 CA-CA pairs (n=1427
# across all native fragments) are bounded below by 4.82 A. We use 3.8
# to leave ~1 A margin, killing only obviously collapsed conformations
# while letting tightly folded near-native shapes pass.
MIN_SEP: float = 3.8

# Number of recent history entries to skip in the clash check. With
# bend window [30°, 110°], distances to ca[k] (= L, fixed at 3.8) and to
# ca[k-1] (= 2 L cos(theta/2) >= 4.36 A) are guaranteed >= MIN_SEP, so
# they can be safely skipped.
SKIP_RECENT: int = 2


def case_reachable(inputs: EncoderInputs, epsilon: float) -> bool:
    """Whole-case precheck: anchor_right reachable from anchor_left
    in n_bonds at maximum stride L_MAX (+ epsilon slack)."""
    d = float(np.linalg.norm(inputs.anchor_right - inputs.anchor_left))
    return d <= inputs.n_bonds * L_MAX + epsilon


def reach_mask(
    p_next_arr: np.ndarray,
    anchor_right: np.ndarray,
    remaining: int,
    epsilon: float,
) -> np.ndarray:
    """Reach upper/lower bound mask for a batch of candidate next CAs.

    Parameters
    ----------
    p_next_arr : (M, 3) candidate next CA positions
    anchor_right : (3,) target endpoint
    remaining : bonds remaining AFTER this step
    epsilon : endpoint tolerance

    Mask logic:
      remaining = 0:   dist <= eps  (this step must land in the eps-ball)
      remaining = 1:   L_MIN - eps <= dist <= L_MAX + eps  (one bond shell)
      remaining >= 2:  dist <= remaining * L_MAX + eps      (no doubling back)
    """
    dist = np.linalg.norm(p_next_arr - anchor_right[None, :], axis=1)
    if remaining == 0:
        return dist <= epsilon
    if remaining == 1:
        return (dist >= L_MIN - epsilon) & (dist <= L_MAX + epsilon)
    return dist <= remaining * L_MAX + epsilon


def clash_mask(
    p_next_arr: np.ndarray,
    ca_history: np.ndarray,
    min_sep: float = MIN_SEP,
    skip_recent: int = SKIP_RECENT,
) -> np.ndarray:
    """Long-range self-clash mask (vectorized over candidates).

    Parameters
    ----------
    p_next_arr : (M, 3) candidate positions
    ca_history : (k, 3) all CAs placed so far, including the current
                 starting CA from which p_next was derived. The last
                 `skip_recent` entries are skipped (they are
                 bend-protected from clashing with p_next).

    Returns
    -------
    (M,) bool — True if candidate has no clash.
    """
    if ca_history.shape[0] <= skip_recent:
        return np.ones(p_next_arr.shape[0], dtype=bool)
    history = ca_history[:-skip_recent]                            # (M', 3)
    diffs = p_next_arr[:, None, :] - history[None, :, :]           # (M, M', 3)
    dists = np.linalg.norm(diffs, axis=2)                          # (M, M')
    return ~((dists < min_sep).any(axis=1))


class ReachableSet:
    """Lightweight feasibility wrapper.

    Replaces the previous strict-DP `ReachableSet` while keeping the
    same construction call site. Pre-built state is minimal: just the
    case-level reach precheck. All actual filtering happens at decode
    time via `feasible_lattice_mask`.

    Public attributes (kept for backward compat with run_sample.py /
    smoke / functional tests):
      case_feasible : bool
      F_sizes       : list[int] — nominal candidate count per level
                       (no precomputation; reports N_DIRECTIONS = 64)
      G_sizes, B_sizes, F : legacy placeholders so existing code paths
                             that reference them still work
    """

    def __init__(
        self,
        inputs: EncoderInputs,
        epsilon: float,
        *,
        # legacy kwargs accepted (and ignored) so old call sites don't break
        voxel_size: Optional[float] = None,
        n_dir_bins: Optional[int] = None,
    ):
        self.inputs = inputs
        self.epsilon = epsilon
        self.case_feasible = case_reachable(inputs, epsilon)

        n_levels = inputs.n_residues
        # Diagnostic placeholders: nominal lattice candidate count per
        # level. Actual feasibility is per-step at decode time.
        self.F_sizes: List[int] = [N_DIRECTIONS] * n_levels
        self.G_sizes: List[int] = list(self.F_sizes)
        self.B_sizes: List[int] = list(self.F_sizes)
        self.F: List[dict] = [dict() for _ in range(n_levels)]

    def feasible_lattice_mask(
        self,
        current_pos: np.ndarray,
        prev_dir: np.ndarray,
        level: int,
        ca_history: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """64-direction mask combining reach and (optional) clash.

        Parameters
        ----------
        current_pos : (3,) the CA at the start of this bond
        prev_dir    : (3,) previous bond direction (unit vector)
        level       : 0-indexed bond number (= number of bonds already
                       placed before this step)
        ca_history  : (level + 1, 3) all CAs placed so far, including
                       current_pos. Optional — when omitted, only reach
                       filter is applied (legacy behaviour).
        """
        lattice = lattice_around(prev_dir)                                  # (64, 3)
        # Decode-time bond length is fixed at CA_CA_LENGTH (3.8). Reach
        # bounds use [L_MIN, L_MAX] for slack — they're decoupled from
        # the actual decode L until the dense-sampling rewrite.
        p_next_arr = current_pos[None, :] + CA_CA_LENGTH * lattice          # (64, 3)
        remaining = self.inputs.n_bonds - level - 1
        mask = reach_mask(
            p_next_arr, self.inputs.anchor_right, remaining, self.epsilon,
        )

        # Last-bond constraint: candidate's bend with -v_right_seed must
        # also lie in [BEND_MIN_DEG, BEND_MAX_DEG]. The lattice annulus
        # only enforces the bend with prev_dir; the chain-continuation
        # bend at residue N-1 is otherwise unconstrained.
        if remaining == 0:
            v_right_target = -self.inputs.v_right_seed
            cos_end = lattice @ v_right_target                              # (64,)
            mask &= (cos_end >= _COS_BEND_MAX) & (cos_end <= _COS_BEND_MIN)

        if ca_history is not None and ca_history.size:
            mask &= clash_mask(p_next_arr, ca_history)
        return mask

    def is_feasible(
        self, p: np.ndarray, d: np.ndarray, level: int,
    ) -> bool:
        """Legacy API kept for compatibility. Always returns True for
        non-trivial levels (no precomputed state to query)."""
        return 0 <= level < self.inputs.n_residues


__all__ = [
    "ReachableSet",
    "L_MIN", "L_MAX", "MIN_SEP", "SKIP_RECENT",
    "case_reachable", "reach_mask", "clash_mask",
]
