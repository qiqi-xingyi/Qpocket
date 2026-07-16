# Author: Yuqi Zhang
"""Stateless geometry term helpers shared by FilterHamiltonian and
FullEnergyScorer.

Each function takes a coords array (n, 3) and returns either a count or
an energy contribution. None of these functions weight the result; the
caller supplies its own λ multipliers.
"""
from __future__ import annotations

from typing import Optional, Tuple

import math

import numpy as np


# ---------------------------------------------------------------------- #
# core helpers                                                           #
# ---------------------------------------------------------------------- #

def pairwise_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Return the (n, n) Euclidean distance matrix for `coords` (n, 3)."""
    diffs = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diffs, axis=-1)


def _gap_pair_indices(
    n: int, gap_min: int, gap_max: Optional[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (I, J) indices i<j with gap_min <= j-i <= gap_max."""
    if n < 2:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    iu, ju = np.triu_indices(n, k=gap_min)
    if gap_max is None:
        return iu, ju
    keep = (ju - iu) <= gap_max
    return iu[keep], ju[keep]


# ---------------------------------------------------------------------- #
# clash counts                                                           #
# ---------------------------------------------------------------------- #

def clash_proxy_count(
    coords: np.ndarray,
    *,
    gap_min: int = 3,
    gap_max: int = 5,
    d_min: float = 3.8,
) -> Tuple[int, int]:
    """Count pairs (i, j) with gap_min <= j-i <= gap_max and ||r_i-r_j|| < d_min.

    Returns (n_clash, n_pairs). n_pairs is the size of the index window
    (used by the optional normalize_terms branch in FilterHamiltonian).
    """
    n = coords.shape[0]
    I, J = _gap_pair_indices(n, gap_min, gap_max)
    if I.size == 0:
        return 0, 0
    d = np.linalg.norm(coords[I] - coords[J], axis=1)
    return int(np.sum(d < d_min)), int(I.size)


def clash_full_count(
    coords: np.ndarray,
    *,
    gap_min: int = 3,
    d_min: float = 3.8,
) -> Tuple[int, int]:
    """Global clash count over all pairs with j-i >= gap_min."""
    n = coords.shape[0]
    I, J = _gap_pair_indices(n, gap_min, gap_max=None)
    if I.size == 0:
        return 0, 0
    d = np.linalg.norm(coords[I] - coords[J], axis=1)
    return int(np.sum(d < d_min)), int(I.size)


# ---------------------------------------------------------------------- #
# contact counts (window-aware)                                          #
# ---------------------------------------------------------------------- #

def contact_indicator_pairs(
    coords: np.ndarray,
    *,
    gap_min: int,
    gap_max: Optional[int],
    d_contact: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (I, J, q) where q is 1 for pairs with distance < d_contact.

    All pairs with gap_min <= j-i <= gap_max are returned (even q=0) so
    callers can apply per-pair weights without re-indexing.
    """
    n = coords.shape[0]
    I, J = _gap_pair_indices(n, gap_min, gap_max)
    if I.size == 0:
        return I, J, np.zeros(0, dtype=np.int64)
    d = np.linalg.norm(coords[I] - coords[J], axis=1)
    q = (d < d_contact).astype(np.int64)
    return I, J, q


# ---------------------------------------------------------------------- #
# Rg                                                                     #
# ---------------------------------------------------------------------- #

def radius_of_gyration(coords: np.ndarray) -> float:
    """Mass-equal Rg in coordinate units."""
    if coords.shape[0] == 0:
        return 0.0
    center = coords.mean(axis=0)
    diffs = coords - center
    return float(math.sqrt(float(np.mean(np.sum(diffs * diffs, axis=1)))))


# ---------------------------------------------------------------------- #
# anchor terms                                                           #
# ---------------------------------------------------------------------- #

def endpoint_distance_penalty(
    coords: np.ndarray,
    anchor_right: Optional[np.ndarray],
    *,
    sigma_anchor: float = 3.8,
    cap: float = 10.0,
) -> Tuple[float, bool]:
    """min(||r_last - anchor_right||^2 / sigma^2, cap). Returns (value, available).

    available=False ⇒ anchor_right was None; value is forced to 0.0.
    """
    if anchor_right is None or coords.shape[0] == 0:
        return 0.0, False
    ar = np.asarray(anchor_right, dtype=np.float64)
    d2 = float(np.sum((coords[-1] - ar) ** 2))
    if sigma_anchor <= 0:
        raise ValueError(f"sigma_anchor must be > 0, got {sigma_anchor}")
    val = d2 / (sigma_anchor * sigma_anchor)
    return min(val, float(cap)), True


def direction_penalty(
    coords: np.ndarray,
    v_right_seed: Optional[np.ndarray],
) -> Tuple[float, bool]:
    """1 - cos(angle) between the last bond direction and -v_right_seed.

    Returns (value, available). available=False if v_right_seed is None,
    coords too short, or either vector degenerate.
    """
    if v_right_seed is None or coords.shape[0] < 2:
        return 0.0, False
    last = coords[-1] - coords[-2]
    last_norm = float(np.linalg.norm(last))
    if last_norm < 1e-12:
        return 0.0, False
    last_unit = last / last_norm
    target = -np.asarray(v_right_seed, dtype=np.float64)
    t_norm = float(np.linalg.norm(target))
    if t_norm < 1e-12:
        return 0.0, False
    target = target / t_norm
    cos_v = float(np.clip(np.dot(last_unit, target), -1.0, 1.0))
    return float(1.0 - cos_v), True


def tail_bend_penalty(
    coords: np.ndarray,
    *,
    tail_len: int,
    bend_min_deg: float,
    bend_max_deg: float,
) -> float:
    """Sum of squared out-of-window tail bend deviations (degrees, scaled).

    Looks at bends at the last `tail_len` interior residues. Each bend
    that falls outside [bend_min_deg, bend_max_deg] contributes the
    squared deviation in radians (so the penalty is finite, well-scaled,
    and zero when in-window).
    """
    n = coords.shape[0]
    if n < 3 or tail_len <= 0:
        return 0.0
    # interior bend angles are at residues 1 .. n-2 (n-2 of them).
    n_bends = n - 2
    if n_bends <= 0:
        return 0.0
    start = max(0, n_bends - int(tail_len))
    bonds = coords[1:] - coords[:-1]
    bond_norms = np.linalg.norm(bonds, axis=1)
    bond_norms = np.where(bond_norms < 1e-12, 1e-12, bond_norms)
    bonds_unit = bonds / bond_norms[:, None]
    cos_b = np.einsum(
        "ij,ij->i", bonds_unit[:-1], bonds_unit[1:],
    )
    cos_b = np.clip(cos_b, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_b))
    selected = angles[start:]
    if selected.size == 0:
        return 0.0
    over = np.maximum(selected - bend_max_deg, 0.0)
    under = np.maximum(bend_min_deg - selected, 0.0)
    dev_deg = over + under
    dev_rad = np.radians(dev_deg)
    return float(np.sum(dev_rad * dev_rad))


# ---------------------------------------------------------------------- #
# convenience aggregator class                                           #
# ---------------------------------------------------------------------- #

class GeometryTerms:
    """Thin OO veneer over the helpers above (kept for symmetry with the
    Hamiltonian classes; pure functions are still the primary surface)."""

    @staticmethod
    def clash_proxy(
        coords: np.ndarray, *, gap_max: int, d_min: float,
    ) -> Tuple[int, int]:
        return clash_proxy_count(
            coords, gap_min=3, gap_max=gap_max, d_min=d_min,
        )

    @staticmethod
    def clash_full(
        coords: np.ndarray, *, d_min: float,
    ) -> Tuple[int, int]:
        return clash_full_count(coords, gap_min=3, d_min=d_min)

    @staticmethod
    def contacts(
        coords: np.ndarray,
        *,
        gap_min: int,
        gap_max: Optional[int],
        d_contact: float,
    ):
        return contact_indicator_pairs(
            coords,
            gap_min=gap_min,
            gap_max=gap_max,
            d_contact=d_contact,
        )

    @staticmethod
    def rg(coords: np.ndarray) -> float:
        return radius_of_gyration(coords)


__all__ = [
    "GeometryTerms",
    "pairwise_distance_matrix",
    "clash_proxy_count",
    "clash_full_count",
    "contact_indicator_pairs",
    "radius_of_gyration",
    "endpoint_distance_penalty",
    "direction_penalty",
    "tail_bend_penalty",
]
