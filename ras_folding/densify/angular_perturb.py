# Author: Yuqi Zhang
"""Angular perturbation primitives for the dense filler.

Two operations:
  perturb_bond_vectors  — small-angle rotation of each bond vector
                          while preserving CA-CA bond length exactly.
  local_ca_rmsd         — unaligned per-residue CA RMSD between two
                          coordinate arrays of identical shape.

Why bond-vector perturbation rather than direct Cartesian noise
---------------------------------------------------------------
Adding Gaussian noise to CA coordinates breaks bond geometry: bond
length drifts to 3.8 ± noise. The encoder / decoder / validity layer
all assume bonds are exactly CA_CA_LENGTH. A small angular wiggle on
each bond preserves bond length and only deforms the trace shape,
which is the relevant local degree of freedom for dense filling
inside a basin.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from ras_folding.utils.constants import CA_CA_LENGTH


# Tolerance on the input bond lengths. The decoder writes
# CA_CA_LENGTH exactly, so any meaningful deviation indicates a
# malformed input — surface it instead of silently rescaling.
_BOND_INPUT_TOL: float = 1e-3


def _random_perpendicular_axis(
    v_unit: np.ndarray,
    rng: np.random.Generator,
    *,
    max_retries: int = 8,
) -> np.ndarray:
    """Return a unit vector perpendicular to v_unit, drawn isotropically.

    Method: sample a Gaussian 3-vector, project out the v_unit component,
    renormalize. Retry if the projection collapses to zero (vanishingly
    rare for an isotropic Gaussian).
    """
    for _ in range(max_retries):
        u = rng.standard_normal(3)
        u_perp = u - float(np.dot(u, v_unit)) * v_unit
        n = float(np.linalg.norm(u_perp))
        if n > 1e-9:
            return u_perp / n
    # Pathological fallback — should never happen with float64 Gaussians.
    fallback = np.array([1.0, 0.0, 0.0]) - v_unit[0] * v_unit
    n = float(np.linalg.norm(fallback))
    return fallback / max(n, 1e-12)


def perturb_bond_vectors(
    coords: np.ndarray,
    sigma_deg: float,
    rng: np.random.Generator,
    *,
    ca_length: float = CA_CA_LENGTH,
    fixed_start: bool = True,
) -> np.ndarray:
    """Apply a small angular wiggle to every bond vector.

    Parameters
    ----------
    coords : (n_residues, 3) float64
    sigma_deg : standard deviation of the rotation angle in degrees
    rng : numpy Generator
    ca_length : target CA-CA bond length (every output bond is renormalised
        to exactly this value).
    fixed_start : if True, coords[0] is preserved exactly and the rest of
        the trace is rebuilt from the perturbed bond vectors.

    Returns
    -------
    coords_new : (n_residues, 3) float64. Independent ndarray — the input
        is NOT mutated.

    Raises
    ------
    ValueError
        if `coords` shape is wrong; if `sigma_deg` is negative; if any
        input bond length deviates from ``ca_length`` by more than
        ``_BOND_INPUT_TOL``.
    """
    if not isinstance(coords, np.ndarray):
        raise ValueError(f"coords must be ndarray, got {type(coords).__name__}")
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"coords must be (n, 3); got shape {tuple(coords.shape)}"
        )
    if sigma_deg < 0:
        raise ValueError(f"sigma_deg must be >= 0; got {sigma_deg}")

    n = coords.shape[0]
    if n < 2:
        # Nothing to perturb — return a copy.
        return coords.astype(np.float64, copy=True)

    bonds = coords[1:] - coords[:-1]
    bond_lens = np.linalg.norm(bonds, axis=1)
    max_dev = float(np.max(np.abs(bond_lens - float(ca_length))))
    if max_dev > _BOND_INPUT_TOL:
        raise ValueError(
            f"input bond length deviates from ca_length={ca_length} by "
            f"{max_dev:.3e} > tol={_BOND_INPUT_TOL}; refusing to perturb"
        )

    sigma_rad = math.radians(float(sigma_deg))
    out = np.empty_like(coords, dtype=np.float64)
    if fixed_start:
        out[0] = coords[0]
    else:
        out[0] = coords[0]  # we still anchor at the first residue

    last = out[0]
    for i in range(n - 1):
        v = bonds[i]
        v_norm = float(np.linalg.norm(v))
        if v_norm < 1e-12:
            raise ValueError(
                f"bond {i} has zero length; cannot define rotation axis"
            )
        v_unit = v / v_norm
        if sigma_rad > 0:
            axis = _random_perpendicular_axis(v_unit, rng)
            angle = float(rng.normal(loc=0.0, scale=sigma_rad))
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            # Rodrigues: axis ⊥ v_unit ⇒ axis · v_unit = 0
            v_unit_rot = v_unit * cos_a + np.cross(axis, v_unit) * sin_a
            # Renormalize defensively against float error
            n_new = float(np.linalg.norm(v_unit_rot))
            v_unit_rot = v_unit_rot / max(n_new, 1e-12)
        else:
            v_unit_rot = v_unit
        last = last + float(ca_length) * v_unit_rot
        out[i + 1] = last

    return out


def local_ca_rmsd(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
) -> float:
    """Unaligned (no-Kabsch) per-residue CA RMSD.

    RMSD = sqrt( mean_i ||a_i - b_i||^2 )

    Both arrays must have the same shape (n, 3). Used inside a single
    basin where the parent and child traces are already aligned by
    construction (both anchored at coords[0]).
    """
    a = np.asarray(coords_a, dtype=np.float64)
    b = np.asarray(coords_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"shape mismatch: {a.shape} vs {b.shape}"
        )
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(
            f"expected (n, 3) coords; got {a.shape}"
        )
    if a.shape[0] == 0:
        return 0.0
    diffs = a - b
    msd = float(np.mean(np.sum(diffs * diffs, axis=1)))
    return float(math.sqrt(max(msd, 0.0)))


__all__ = [
    "perturb_bond_vectors",
    "local_ca_rmsd",
]
