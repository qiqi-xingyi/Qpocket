# Author: Yuqi Zhang
"""Per-bond annulus lattice on the unit sphere.

Each bond's 64-direction lattice is constructed dynamically based on the
previous bond's direction. The lattice covers the spherical annulus
where the bend angle (between previous and current bond direction) is
in [BEND_MIN_DEG, BEND_MAX_DEG] = [30°, 110°] — tight enough to encode
the bend constraint as the lattice itself (every candidate is bend-legal
by construction, so no separate bend filter is needed).

Implementation: precompute one canonical lattice (annulus around +z),
then rotate to align +z with the previous bond direction at decode time.
"""
from __future__ import annotations

import math

import numpy as np

# Hard bend constraint range. Tightened on 2026-04-27 from [10°, 180°]
# after measuring native KRAS distribution across 35 fragments (392
# bends): all natives fall in [35.7°, 94.1°]. The new window [30°, 110°]
# - keeps 100% of native bends (>5° lower buffer, ~16° upper buffer)
# - excludes the empty [10°, 30°] tail that was wasting lattice candidates
# - upper bound 110° is well below the 133° self-intersection threshold
#   (where ||CA[k+1] - CA[k-1]|| would drop below ~3 Å for L = 3.8)
BEND_MIN_DEG: float = 30.0
BEND_MAX_DEG: float = 110.0
N_DIRECTIONS: int = 64

_COS_BEND_MIN: float = math.cos(math.radians(BEND_MIN_DEG))   # ≈ 0.866
_COS_BEND_MAX: float = math.cos(math.radians(BEND_MAX_DEG))   # ≈ -0.342


def build_canonical_lattice(n: int = N_DIRECTIONS) -> np.ndarray:
    """Build canonical lattice: n unit vectors on annulus around +z axis.

    Polar angle θ (from +z) ∈ [BEND_MIN_DEG, BEND_MAX_DEG], i.e.
    z = cos(θ) ∈ [_COS_BEND_MAX, _COS_BEND_MIN].

    Azimuthal angle: golden-angle Fibonacci spiral.

    Returns
    -------
    (n, 3) float64 unit vectors.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2; got {n}")
    phi_golden = math.pi * (3.0 - math.sqrt(5.0))
    out = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        # Linearly map index to z in [z_max, z_min] (top of annulus to bottom)
        # i=0 -> z = _COS_BEND_MIN (small θ ≈ 50°)
        # i=n-1 -> z = _COS_BEND_MAX (large θ ≈ 160°)
        t = i / (n - 1)
        z = _COS_BEND_MIN - t * (_COS_BEND_MIN - _COS_BEND_MAX)
        r = math.sqrt(max(0.0, 1.0 - z * z))
        theta_az = i * phi_golden
        x = r * math.cos(theta_az)
        y = r * math.sin(theta_az)
        out[i] = (x, y, z)
    # Renormalize for numerical safety
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    out = out / (norms + 1e-12)
    return out


# Module-level: build once at import.
_CANONICAL_LATTICE: np.ndarray = build_canonical_lattice()


def _rotation_matrix_z_to(target: np.ndarray) -> np.ndarray:
    """Return 3×3 rotation that maps +z onto `target` (unit-vector aligned).

    Uses Rodrigues' formula. Special-cases target ≈ ±z.
    """
    z_axis = np.array([0.0, 0.0, 1.0])
    t = target / (np.linalg.norm(target) + 1e-12)
    cos_a = float(np.dot(z_axis, t))

    if cos_a > 1.0 - 1e-9:
        return np.eye(3)
    if cos_a < -1.0 + 1e-9:
        # 180° about x-axis (flips +z to -z, leaves x unchanged)
        return np.array([[1.0,  0.0,  0.0],
                          [0.0, -1.0,  0.0],
                          [0.0,  0.0, -1.0]])

    axis = np.cross(z_axis, t)
    sin_a = float(np.linalg.norm(axis))
    axis = axis / sin_a
    K = np.array([[0.0,     -axis[2], axis[1]],
                   [axis[2],  0.0,    -axis[0]],
                   [-axis[1], axis[0], 0.0]])
    return np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)


def lattice_around(prev_dir: np.ndarray) -> np.ndarray:
    """Return the per-bond 64-direction lattice given the previous bond direction.

    Each direction d_i in the result satisfies:
      bend angle (between prev_dir and d_i) ∈ [BEND_MIN_DEG, BEND_MAX_DEG]

    Parameters
    ----------
    prev_dir : (3,) unit vector — direction of the previous bond
               (or v_left_seed for bond 0).

    Returns
    -------
    (64, 3) float64 unit vectors.
    """
    prev_dir = np.asarray(prev_dir, dtype=np.float64)
    R = _rotation_matrix_z_to(prev_dir)
    # canonical[i] = (x, y, z) → R @ canonical[i] places +z onto prev_dir
    return _CANONICAL_LATTICE @ R.T


__all__ = [
    "BEND_MIN_DEG",
    "BEND_MAX_DEG",
    "N_DIRECTIONS",
    "build_canonical_lattice",
    "lattice_around",
]
