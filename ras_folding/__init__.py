# Author: Yuqi Zhang
# --*-- conding:utf-8 --*--
# @time:4/27/26 4:39 PM
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:__init__.py.py
"""ras_folding — autoregressive CA-trace encoder for KRAS pocket fragment conformational sampling.

Public API (re-exported from ras_folding.encoder):
  EncoderInputs        : geometric inputs dataclass (sequence-derived)
  decode_bitstring     : bitstring + inputs → (n_residues, 3) CA trace
  ReachableSet         : reach + clash feasibility filter
  build_canonical_lattice / lattice_around : annulus lattice utilities
  Constants            : EPSILON, BITS_PER_BOND, BEND_MIN_DEG, BEND_MAX_DEG,
                          N_DIRECTIONS, L_MIN, L_MAX, MIN_SEP, SKIP_RECENT
"""
from ras_folding.encoder import (
    BEND_MAX_DEG,
    BEND_MIN_DEG,
    BITS_PER_BOND,
    EPSILON,
    EncoderInputs,
    L_MAX,
    L_MIN,
    MIN_SEP,
    N_DIRECTIONS,
    ReachableSet,
    SKIP_RECENT,
    build_canonical_lattice,
    decode_bitstring,
    lattice_around,
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
