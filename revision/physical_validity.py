#!/usr/bin/env python3
# Author: Yuqi Zhang
"""Validity for structures that did not come from the lattice decoder.

The production battery mixes two kinds of check. Some are assertions that
the decoder did its job -- a CA-CA bond within one micro-angstrom of
3.80, an endpoint sitting exactly on the anchor, a final direction
matching the seed -- and its own documentation says as much: the decoder
writes ``CA_CA_LENGTH * unit_vec`` exactly, so anything else is a bug in
the decoder rather than a bad conformation. The rest are statements about
the structure: residues far apart in sequence must not occupy the same
space, and the fragment must still reach its anchor.

Applied to a loop remodeller's output the first kind rejects everything,
because real backbones have CA-CA distances that vary by a few hundredths
of an angstrom. Rejecting those structures would be rejecting them for
not being lattice encodings, which is rejecting them for not being the
method under test.

This module keeps the structural checks and replaces the decoder
assertions with physical tolerances, so a comparator's structure is
refused for the reasons a structure can be bad and not for the reasons a
decoder can be broken. The tolerances are stated here rather than tuned:
the bond-length window covers observed CA-CA geometry, and the clash and
endpoint criteria are the production values unchanged.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ras_folding.encoder.reachable import MIN_SEP
from ras_folding.sampler.context import get_encoder_inputs
from ras_folding.sampler.validity import EPSILON_ENDPOINT_A
from ras_folding.utils.constants import CA_CA_LENGTH

# Consecutive CA distances in solved structures sit near 3.80 A for trans
# peptides and near 2.9 A for the rare cis ones. The window admits the
# former with room for ordinary deviation and still excludes a chain that
# has been torn apart.
BOND_TOL_PHYSICAL_A = 0.30


def validate_physical_coords(
    coords: Optional[np.ndarray], context_or_inputs,
    bond_tol: float = BOND_TOL_PHYSICAL_A,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Structural validity for a coordinate trace of any provenance.

    Returns the same ``(valid, reason, info)`` shape as the production
    battery, and the reasons carry the same names where the check is the
    same one, so results from both paths can be pooled without a
    translation table.
    """
    info: Dict[str, Any] = {"validator": "physical",
                            "bond_tol_A": float(bond_tol)}
    if coords is None:
        return False, "decode_failed", info
    coords = np.asarray(coords, dtype=np.float64)

    ei = get_encoder_inputs(context_or_inputs)
    n = int(getattr(ei, "n_residues", 0))
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] != n:
        info["coords_shape"] = tuple(coords.shape)
        info["expected_shape"] = (n, 3)
        return False, "bad_shape", info
    if not np.all(np.isfinite(coords)):
        return False, "nan_coords", info

    if n >= 2:
        lens = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
        dev = float(np.max(np.abs(lens - CA_CA_LENGTH)))
        info.update({"max_bond_dev": dev,
                     "bond_len_min": float(lens.min()),
                     "bond_len_max": float(lens.max())})
        if dev > bond_tol:
            return False, "bond_length_violation", info

    # Unchanged from the production battery: a trace that folds through
    # itself is invalid whatever produced it.
    if n >= 4:
        d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        i = np.arange(n)[:, None]
        j = np.arange(n)[None, :]
        far = (j - i) >= 3
        if far.any():
            lr = float(d[far].min())
            info["long_range_min_dist"] = lr
            if lr < MIN_SEP:
                return False, "long_range_clash", info

    # Also unchanged: the fragment has to still reach the anchor it was
    # built between, which is a property of the structure and not of the
    # encoding.
    anchor_right = getattr(ei, "anchor_right", None)
    if anchor_right is not None and n >= 1:
        dist = float(np.linalg.norm(
            coords[-1] - np.asarray(anchor_right, dtype=np.float64)))
        info["endpoint_dist"] = dist
        if dist > EPSILON_ENDPOINT_A:
            return False, "endpoint_mismatch", info

    return True, None, info


__all__ = ["validate_physical_coords", "BOND_TOL_PHYSICAL_A"]
