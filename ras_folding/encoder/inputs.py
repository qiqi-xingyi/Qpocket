# Author: Yuqi Zhang
"""Encoder inputs — geometric inputs needed for autoregressive decoding.

Single-direction walk goes from anchor_left to anchor_right with N-1
bonds. Each bond's lattice depends on the previous bond's direction,
which is itself selected from the lattice (autoregressive).

The two seed directions (v_left_seed, v_right_seed) anchor the first
and last bond's bend constraints to the parent protein flanks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EncoderInputs:
    """Geometric inputs for autoregressive single-direction walk.

    Convention
    ----------
    Walk direction: N -> C (matches PDB residue order).
    anchor_left  = fragment_ca_ref[0]   (= first fragment CA, given)
    anchor_right = fragment_ca_ref[-1]  (= last fragment CA, given)

    v_left_seed: chain-forward direction of the bond from flank_before[-1]
                  into anchor_left. Used as the "previous bond" for bond 0's
                  lattice axis. Constrains bend at residue 0.

    v_right_seed: chain-reverse direction of the bond from flank_after[0]
                   into anchor_right. Used as the "next bond" reference for
                   bond (N-2)'s last-bend constraint at residue N-1.

    Both seed vectors are unit vectors.
    """
    n_residues: int
    anchor_left: np.ndarray   # (3,) float64
    anchor_right: np.ndarray  # (3,) float64
    v_left_seed: np.ndarray   # (3,) float64, unit
    v_right_seed: np.ndarray  # (3,) float64, unit

    @property
    def n_bonds(self) -> int:
        return self.n_residues - 1

    @property
    def n_bonds_left(self) -> int:
        """Left half's bond count for bidirectional walk: ceil(n_bonds / 2).

        Convention: left side gets the extra bond when n_bonds is odd, so
        the meeting CA index = n_bonds_left from the left.
        """
        return (self.n_bonds + 1) // 2

    @property
    def n_bonds_right(self) -> int:
        """Right half's bond count: floor(n_bonds / 2)."""
        return self.n_bonds // 2

    @property
    def meeting_index(self) -> int:
        """CA index where the two walks meet (= n_bonds_left)."""
        return self.n_bonds_left

    @classmethod
    def from_fragment_context(cls, ctx) -> "EncoderInputs":
        """Build from a FragmentContext.

        Per E4: when flank_before is empty, fall back to native first bond
        direction (fragment_ca_ref[1] - fragment_ca_ref[0]) as v_left_seed.
        Symmetrically for flank_after.
        """
        n_residues = int(ctx.n_residues)
        if n_residues < 3:
            raise ValueError(
                f"EncoderInputs requires n_residues >= 3, got {n_residues}"
            )

        anchor_left = np.asarray(ctx.fragment_ca_ref[0], dtype=np.float64)
        anchor_right = np.asarray(ctx.fragment_ca_ref[-1], dtype=np.float64)

        # v_left_seed: flank_before[-1] -> anchor_left (chain forward)
        if ctx.flank_before_ca.size:
            v_l = anchor_left - np.asarray(ctx.flank_before_ca[-1], dtype=np.float64)
        else:
            # Fallback: native first bond direction (information leak; one
            # KRAS case 9IAY hits this path)
            v_l = (np.asarray(ctx.fragment_ca_ref[1], dtype=np.float64)
                    - anchor_left)
        v_l_norm = np.linalg.norm(v_l)
        if v_l_norm < 1e-9:
            raise ValueError(
                "v_left_seed degenerate (flank coincides with anchor_left)"
            )
        v_left_seed = v_l / v_l_norm

        # v_right_seed: flank_after[0] -> anchor_right (chain reversed,
        # per D3). Equivalently: anchor_right - flank_after[0], not negated.
        if ctx.flank_after_ca.size:
            v_r = anchor_right - np.asarray(ctx.flank_after_ca[0], dtype=np.float64)
        else:
            # Fallback: native last bond direction reversed
            v_r = (anchor_right
                    - np.asarray(ctx.fragment_ca_ref[-2], dtype=np.float64))
        v_r_norm = np.linalg.norm(v_r)
        if v_r_norm < 1e-9:
            raise ValueError(
                "v_right_seed degenerate (flank coincides with anchor_right)"
            )
        v_right_seed = v_r / v_r_norm

        return cls(
            n_residues=n_residues,
            anchor_left=anchor_left,
            anchor_right=anchor_right,
            v_left_seed=v_left_seed,
            v_right_seed=v_right_seed,
        )


__all__ = ["EncoderInputs"]
