# Author: Yuqi Zhang
"""IDE-runnable I/O smoke test for ras_folding.encoder.

Purpose
-------
Exercise the public input/output contract of `decode_bitstring` and
`EncoderInputs` without depending on any external case loader. Two
synthetic cases are constructed in-script:

  case A (feasible)   : anchor_right placed within n_bonds * L_MAX of
                        anchor_left → walk should land within EPSILON.
  case B (infeasible) : anchor_right placed far beyond n_bonds * L_MAX
                        → triggers the fallback path; endpoint residual
                        will exceed EPSILON.

For each case the script prints:
  - input shapes / dtypes / values
  - output shape / dtype
  - endpoint residual (||ca[-1] - anchor_right||)
  - bond-length min/max/mean (should all be CA_CA_LENGTH = 3.8)
  - bend-angle min/max (should fall in [BEND_MIN_DEG, BEND_MAX_DEG]
    when no fallback step is taken)
  - long-range pairwise CA-CA min distance (skip i,i+1, i,i+2)
  - ReachableSet.case_feasible flag

Run
---
From the project root:
    python ras_folding/test_io.py
or in an IDE: open this file and run it directly. The script prepends
the project root to sys.path so the package import resolves whether or
not ras_folding is installed.

This script does NOT mutate any project state and does NOT write files.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Tuple

import numpy as np

# --- path bootstrap: allow running this file directly from an IDE -------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ras_folding.encoder import (  # noqa: E402
    BEND_MAX_DEG,
    BEND_MIN_DEG,
    BITS_PER_BOND,
    EPSILON,
    EncoderInputs,
    L_MAX,
    L_MIN,
    MIN_SEP,
    ReachableSet,
    decode_bitstring,
)
from ras_folding.utils.constants import CA_CA_LENGTH  # noqa: E402


# ---------------------------------------------------------------------- #
# Helpers                                                                #
# ---------------------------------------------------------------------- #

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("cannot normalize zero vector")
    return v / n


def _make_inputs(
    *,
    n_residues: int,
    anchor_right_offset: np.ndarray,
    v_left_seed: np.ndarray,
    v_right_seed: np.ndarray,
) -> EncoderInputs:
    """Build an EncoderInputs at the origin with anchor_right offset."""
    anchor_left = np.zeros(3, dtype=np.float64)
    anchor_right = anchor_left + np.asarray(anchor_right_offset, dtype=np.float64)
    return EncoderInputs(
        n_residues=int(n_residues),
        anchor_left=anchor_left,
        anchor_right=anchor_right,
        v_left_seed=_unit(np.asarray(v_left_seed, dtype=np.float64)),
        v_right_seed=_unit(np.asarray(v_right_seed, dtype=np.float64)),
    )


def _bond_lengths(ca: np.ndarray) -> np.ndarray:
    return np.linalg.norm(ca[1:] - ca[:-1], axis=1)


def _bend_angles_deg(ca: np.ndarray) -> np.ndarray:
    """Bend angle (degrees) at residues 1..N-2 between consecutive bonds."""
    if ca.shape[0] < 3:
        return np.zeros(0, dtype=np.float64)
    d = ca[1:] - ca[:-1]
    d_norm = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    cos_b = np.einsum("ij,ij->i", d_norm[:-1], d_norm[1:])
    cos_b = np.clip(cos_b, -1.0, 1.0)
    return np.degrees(np.arccos(cos_b))


def _long_range_min_dist(ca: np.ndarray, gap: int = 3) -> float:
    """Min pairwise CA-CA distance for |i - j| >= gap. Returns +inf if none."""
    n = ca.shape[0]
    if n < gap + 1:
        return float("inf")
    diffs = ca[:, None, :] - ca[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    iu = np.arange(n)[:, None]
    ju = np.arange(n)[None, :]
    mask = (ju - iu) >= gap
    if not mask.any():
        return float("inf")
    return float(dists[mask].min())


def _format_vec(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:+.4f}" for x in v) + "]"


# ---------------------------------------------------------------------- #
# Single-case probe                                                      #
# ---------------------------------------------------------------------- #

def probe_case(
    label: str,
    inputs: EncoderInputs,
    bitstrings: Tuple[int, ...],
) -> None:
    print("=" * 72)
    print(f"CASE {label}")
    print("=" * 72)

    # --- input report --------------------------------------------------
    print("[inputs]")
    print(f"  n_residues       = {inputs.n_residues}")
    print(f"  n_bonds          = {inputs.n_bonds}")
    print(f"  anchor_left      = {_format_vec(inputs.anchor_left)}  "
          f"shape={inputs.anchor_left.shape}  dtype={inputs.anchor_left.dtype}")
    print(f"  anchor_right     = {_format_vec(inputs.anchor_right)}  "
          f"shape={inputs.anchor_right.shape}  dtype={inputs.anchor_right.dtype}")
    print(f"  v_left_seed      = {_format_vec(inputs.v_left_seed)}  "
          f"|v|={np.linalg.norm(inputs.v_left_seed):.6f}")
    print(f"  v_right_seed     = {_format_vec(inputs.v_right_seed)}  "
          f"|v|={np.linalg.norm(inputs.v_right_seed):.6f}")
    anchor_dist = float(np.linalg.norm(inputs.anchor_right - inputs.anchor_left))
    print(f"  ||A_R - A_L||    = {anchor_dist:.4f}  "
          f"(reach upper bound = n_bonds * L_MAX + ε = "
          f"{inputs.n_bonds * L_MAX + EPSILON:.4f})")

    # --- reachable precheck --------------------------------------------
    rs = ReachableSet(inputs, EPSILON)
    print(f"[ReachableSet] case_feasible = {rs.case_feasible}")

    # --- decode each bitstring -----------------------------------------
    n_bits = inputs.n_bonds * BITS_PER_BOND
    full_mask = (1 << n_bits) - 1 if n_bits > 0 else 0
    print(f"[bitstring layout]")
    print(f"  bits_per_bond    = {BITS_PER_BOND}")
    print(f"  total_bits       = {n_bits}")
    print(f"  bitstring range  = [0, 2^{n_bits} - 1]  (mask=0x{full_mask:x})")

    for bs in bitstrings:
        bs_clipped = bs & full_mask if n_bits > 0 else 0
        print("-" * 72)
        print(f"[decode] bitstring = {bs_clipped}  (binary, MSB-first):")
        print(f"         0b{bs_clipped:0{max(n_bits, 1)}b}")

        ca = decode_bitstring(bs_clipped, inputs, epsilon=EPSILON, reachable=rs)

        # --- output report --------------------------------------------
        print(f"[output] type={type(ca).__name__}  shape={ca.shape}  "
              f"dtype={ca.dtype}")

        # anchor_left exactness
        al_err = float(np.linalg.norm(ca[0] - inputs.anchor_left))
        print(f"  ||ca[0]  - anchor_left||  = {al_err:.3e}  "
              f"(should be 0 by construction)")

        # endpoint residual
        ar_err = float(np.linalg.norm(ca[-1] - inputs.anchor_right))
        within_eps = ar_err <= EPSILON
        print(f"  ||ca[-1] - anchor_right|| = {ar_err:.4f}  "
              f"(EPSILON={EPSILON})  within_eps={within_eps}")

        # bond length distribution
        bl = _bond_lengths(ca)
        print(f"  bond lengths  : min={bl.min():.4f}  max={bl.max():.4f}  "
              f"mean={bl.mean():.4f}  (target CA_CA_LENGTH={CA_CA_LENGTH})")

        # bend angle distribution
        bends = _bend_angles_deg(ca)
        if bends.size:
            in_window = ((bends >= BEND_MIN_DEG) & (bends <= BEND_MAX_DEG)).all()
            print(f"  bend angles   : min={bends.min():.2f}°  "
                  f"max={bends.max():.2f}°  "
                  f"all_in[{BEND_MIN_DEG},{BEND_MAX_DEG}]={in_window}")
        else:
            print(f"  bend angles   : <none> (n_residues<3)")

        # long-range clash diagnostic
        lr_min = _long_range_min_dist(ca, gap=3)
        clash_free = lr_min >= MIN_SEP if math.isfinite(lr_min) else True
        print(f"  long-range min CA-CA (|i-j|>=3) = {lr_min:.4f}  "
              f"(MIN_SEP={MIN_SEP})  clash_free={clash_free}")

        # last-bond bend vs v_right_seed
        if inputs.n_bonds >= 1:
            last_dir = (ca[-1] - ca[-2])
            last_dir = last_dir / (np.linalg.norm(last_dir) + 1e-12)
            cos_end = float(np.dot(last_dir, -inputs.v_right_seed))
            cos_end_clipped = max(-1.0, min(1.0, cos_end))
            ang_end = math.degrees(math.acos(cos_end_clipped))
            in_window_end = (BEND_MIN_DEG <= ang_end <= BEND_MAX_DEG)
            print(f"  last-bond vs -v_right_seed bend = {ang_end:.2f}°  "
                  f"in[{BEND_MIN_DEG},{BEND_MAX_DEG}]={in_window_end}")
    print()


# ---------------------------------------------------------------------- #
# Main                                                                   #
# ---------------------------------------------------------------------- #

def main() -> None:
    print(f"ras_folding I/O smoke test")
    print(f"  CA_CA_LENGTH = {CA_CA_LENGTH}")
    print(f"  L_MIN,L_MAX  = {L_MIN}, {L_MAX}")
    print(f"  BEND window  = [{BEND_MIN_DEG}, {BEND_MAX_DEG}] deg")
    print(f"  MIN_SEP      = {MIN_SEP}")
    print(f"  EPSILON      = {EPSILON}")
    print(f"  BITS_PER_BOND= {BITS_PER_BOND}")
    print()

    # --- case A: feasible ------------------------------------------------
    # n_residues=8 → n_bonds=7; place anchor_right at distance ~ n_bonds * 2.0
    # so any reasonable walk can reach it well within n_bonds * L_MAX.
    n_A = 8
    target_dist_A = n_A * 1.8  # 8 residues, 7 bonds, comfortably reachable
    inputs_A = _make_inputs(
        n_residues=n_A,
        anchor_right_offset=np.array([target_dist_A, 0.0, 0.0]),
        v_left_seed=np.array([1.0, 0.0, 0.0]),
        v_right_seed=np.array([1.0, 0.0, 0.0]),
    )
    # bitstring set: all-zeros, all-ones-mask, a striped pattern, a "random" int.
    n_bits_A = inputs_A.n_bonds * BITS_PER_BOND
    bs_A = (
        0,
        (1 << n_bits_A) - 1,
        int("010101" * inputs_A.n_bonds, 2),
        0xC0FFEE & ((1 << n_bits_A) - 1),
    )
    probe_case("A_feasible", inputs_A, bs_A)

    # --- case B: infeasible (forces fallback) ----------------------------
    # anchor_right placed beyond n_bonds * L_MAX → case_feasible=False; the
    # autoregressive walk will hit valid_indices=∅ at some step and trigger
    # the fallback branch in decoder.py.
    n_B = 6
    far_dist_B = n_B * L_MAX * 5.0  # well beyond reachable
    inputs_B = _make_inputs(
        n_residues=n_B,
        anchor_right_offset=np.array([far_dist_B, 0.0, 0.0]),
        v_left_seed=np.array([1.0, 0.0, 0.0]),
        v_right_seed=np.array([1.0, 0.0, 0.0]),
    )
    n_bits_B = inputs_B.n_bonds * BITS_PER_BOND
    bs_B = (
        0,
        (1 << n_bits_B) - 1,
    )
    probe_case("B_infeasible_fallback", inputs_B, bs_B)

    print("done.")


if __name__ == "__main__":
    main()
