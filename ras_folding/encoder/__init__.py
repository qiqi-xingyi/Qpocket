# Author: Yuqi Zhang
"""ras_folding encoder — bitstring ↔ CA trace.

Public API:
  EncoderInputs        : geometric inputs dataclass (sequence-derived)
  decode_bitstring     : bitstring + inputs → (n_residues, 3) CA trace
  build_canonical_lattice / lattice_around : annulus lattice utilities
  EPSILON, BEND_MIN_DEG, BEND_MAX_DEG, N_DIRECTIONS : constants
"""
from ras_folding.encoder.inputs import EncoderInputs
from ras_folding.encoder.lattice import (
    BEND_MAX_DEG,
    BEND_MIN_DEG,
    N_DIRECTIONS,
    build_canonical_lattice,
    lattice_around,
)
from ras_folding.encoder.decoder import (
    BITS_PER_BOND,
    EPSILON,
    decode_bitstring,
)
from ras_folding.encoder.reachable import (
    L_MAX,
    L_MIN,
    MIN_SEP,
    SKIP_RECENT,
    ReachableSet,
)

__all__ = [
    "EncoderInputs",
    "decode_bitstring",
    "ReachableSet",
    "EPSILON",
    "BITS_PER_BOND",
    "BEND_MIN_DEG",
    "BEND_MAX_DEG",
    "N_DIRECTIONS",
    "L_MIN",
    "L_MAX",
    "MIN_SEP",
    "SKIP_RECENT",
    "build_canonical_lattice",
    "lattice_around",
]
