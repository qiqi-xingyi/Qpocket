# Author: Yuqi Zhang
"""Autoregressive decoder: bitstring → (n_residues, 3) CA trace.

Single-direction walk from anchor_left to anchor_right with strict
DP-based reachability pruning. The decoder is built around a precomputed
`ReachableSet` for the case; lattice candidates that don't lead to a
valid completion are removed before the bitstring's 6-bit code selects.

Decode procedure (per bond k = 0 .. N-2):
  1. Build full 64-lattice around d_{k-1} (= v_left_seed for k=0).
  2. Compute reachability mask via ReachableSet: which lattice indices
     lead to a state in F[k+1].
  3. From the bitstring's 6 bits for this bond, modulo into valid set
     to pick the actual lattice index.
  4. Step: CA[k+1] = CA[k] + 3.8 × lattice[chosen_idx].

Output: (n_residues, 3) array.
  - CA[0] = anchor_left exactly (anchored).
  - CA[1..N-2] from the walk.
  - CA[N-1] is the walked endpoint (within ε of anchor_right when the
    case is feasible and the bitstring's encoded path was non-empty
    valid set at every step).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ras_folding.encoder.inputs import EncoderInputs
from ras_folding.encoder.lattice import (
    BEND_MAX_DEG,
    BEND_MIN_DEG,
    N_DIRECTIONS,
    lattice_around,
)
from ras_folding.encoder.reachable import ReachableSet
from ras_folding.utils.constants import CA_CA_LENGTH

# Endpoint-match tolerance (Å). Fixed value per user decision (容差先给一个固定值).
EPSILON: float = 1.0

BITS_PER_BOND: int = 6


def _bitstring_to_codes(bitstring: int, n_bonds: int) -> np.ndarray:
    """Slice a bitstring integer into n_bonds 6-bit codes (MSB-first).

    Bond 0's code occupies the most-significant 6 bits, bond N-2's the
    least-significant. Each code ∈ [0, 63].
    """
    mask = (1 << BITS_PER_BOND) - 1
    codes = np.zeros(n_bonds, dtype=np.int64)
    for i in range(n_bonds):
        shift = (n_bonds - 1 - i) * BITS_PER_BOND
        codes[i] = (bitstring >> shift) & mask
    return codes


def decode_bitstring_with_info(
    bitstring: int,
    inputs: EncoderInputs,
    *,
    epsilon: float = EPSILON,
    reachable: Optional[ReachableSet] = None,
) -> "tuple[np.ndarray, dict]":
    """Like decode_bitstring but also returns a per-step info dict.

    Returns
    -------
    coords : (n_residues, 3) float64 — same as decode_bitstring.
    info : dict with keys
        valid              : bool
        fallback_triggered : bool
        fallback_steps     : list[int]   — bond-step indices that hit
                              valid_indices.size == 0
        n_fallback_steps   : int
        endpoint_residual  : float       — ||ca[-1] - anchor_right||,
                              or None if anchor_right unavailable

    The ``valid`` flag is True iff:
      - no fallback was triggered, AND
      - endpoint_residual is None (no anchor) OR endpoint_residual <= epsilon.

    This function is the canonical instrumented decoder. The original
    ``decode_bitstring`` (below) is unchanged and returns coords only,
    preserving the existing public interface.
    """
    if reachable is None:
        reachable = ReachableSet(inputs, epsilon)

    n = inputs.n_residues
    n_bonds = inputs.n_bonds
    codes = _bitstring_to_codes(bitstring, n_bonds)

    ca = np.zeros((n, 3), dtype=np.float64)
    ca[0] = inputs.anchor_left
    last_dir = inputs.v_left_seed.copy()
    fallback_steps: list = []

    for k in range(n_bonds):
        lattice = lattice_around(last_dir)
        mask = reachable.feasible_lattice_mask(
            ca[k], last_dir, k, ca_history=ca[: k + 1],
        )
        valid_indices = np.flatnonzero(mask)

        if valid_indices.size > 0:
            chosen = int(valid_indices[int(codes[k]) % valid_indices.size])
        else:
            fallback_steps.append(k)
            target = inputs.anchor_right - ca[k]
            target_norm = np.linalg.norm(target)
            if target_norm > 1e-9:
                target = target / target_norm
                similarities = lattice @ target
                chosen = int(np.argmax(similarities))
            else:
                chosen = int(int(codes[k]) % N_DIRECTIONS)

        d = lattice[chosen]
        ca[k + 1] = ca[k] + CA_CA_LENGTH * d
        last_dir = d

    anchor_right = getattr(inputs, "anchor_right", None)
    if anchor_right is not None and n >= 1:
        endpoint_residual = float(np.linalg.norm(ca[-1] - anchor_right))
    else:
        endpoint_residual = None

    fallback_triggered = bool(fallback_steps)
    valid = (not fallback_triggered) and (
        endpoint_residual is None or endpoint_residual <= epsilon
    )

    info = {
        "valid": valid,
        "fallback_triggered": fallback_triggered,
        "fallback_steps": list(fallback_steps),
        "n_fallback_steps": len(fallback_steps),
        "endpoint_residual": endpoint_residual,
    }
    return ca, info


def decode_bitstring(
    bitstring: int,
    inputs: EncoderInputs,
    *,
    epsilon: float = EPSILON,
    reachable: Optional[ReachableSet] = None,
) -> np.ndarray:
    """Decode bitstring into a CA trace via single-direction autoregressive walk.

    Parameters
    ----------
    bitstring  : int — the encoded path. Length = n_bonds × 6 bits.
    inputs     : EncoderInputs for this case.
    epsilon    : endpoint tolerance (Å). Used both for `reachable` build
                 and for fallback feasibility heuristics.
    reachable  : optional precomputed ReachableSet. If None, builds one
                 (expensive; share across many decodes for the same case).

    Returns
    -------
    ca : (n_residues, 3) float64 CA trace. CA[0] = anchor_left exactly.
    """
    if reachable is None:
        reachable = ReachableSet(inputs, epsilon)

    n = inputs.n_residues
    n_bonds = inputs.n_bonds
    codes = _bitstring_to_codes(bitstring, n_bonds)

    ca = np.zeros((n, 3), dtype=np.float64)
    ca[0] = inputs.anchor_left
    last_dir = inputs.v_left_seed.copy()

    for k in range(n_bonds):
        lattice = lattice_around(last_dir)  # (64, 3)
        # ca[: k + 1] are all CAs placed so far including current ca[k].
        # Passing the history activates the long-range clash filter.
        mask = reachable.feasible_lattice_mask(
            ca[k], last_dir, k, ca_history=ca[: k + 1],
        )
        valid_indices = np.flatnonzero(mask)

        if valid_indices.size > 0:
            chosen = int(valid_indices[int(codes[k]) % valid_indices.size])
        else:
            # No DP-feasible candidate at this step. This means the walk
            # has wandered into a region from which the case-level
            # ReachableSet was already filtered out — i.e., we are in a
            # leaf where no valid completion exists.
            #
            # Fallback: greedy step toward anchor_right. This produces an
            # invalid (non-completable) walk but lets us still return a
            # full (n_residues, 3) array for inspection / debugging. The
            # endpoint will not match anchor_right within ε.
            target = inputs.anchor_right - ca[k]
            target_norm = np.linalg.norm(target)
            if target_norm > 1e-9:
                target = target / target_norm
                similarities = lattice @ target
                chosen = int(np.argmax(similarities))
            else:
                chosen = int(int(codes[k]) % N_DIRECTIONS)

        d = lattice[chosen]
        ca[k + 1] = ca[k] + CA_CA_LENGTH * d
        last_dir = d

    return ca


__all__ = [
    "EPSILON",
    "BITS_PER_BOND",
    "decode_bitstring",
    "decode_bitstring_with_info",
]
