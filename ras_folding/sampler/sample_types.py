# Author: Yuqi Zhang
"""Sample-level dataclasses and bitstring/code conversion utilities.

Two dataclasses are exposed:
  CandidateSample : one decoded / scored candidate
  SampleBatch     : a collection of CandidateSamples produced at a single
                     tau (or for a single base-sampler call), plus
                     aggregate statistics

Bitstring representation
------------------------
The encoder's ``decode_bitstring`` (ras_folding/encoder/decoder.py:60) takes
an ``int``. To stay aligned with the spec ("bitstring: str | None"), we
store the canonical form in CandidateSample as a string of '0' and '1'
characters of length ``n_bonds * BITS_PER_BOND``. Conversion utilities
below convert between str, int, and per-bond code arrays. They are pure
and bit-exact w.r.t. the decoder's MSB-first convention
(decoder.py:46-57). NOTE: this module does NOT call the decoder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ras_folding.encoder.decoder import BITS_PER_BOND


# ---------------------------------------------------------------------- #
# bitstring / codes utilities                                            #
# ---------------------------------------------------------------------- #

def codes_to_bitstring_int(codes: Sequence[int], n_bonds: int) -> int:
    """Pack `n_bonds` 6-bit codes into a single int, MSB-first.

    codes[0] occupies the most-significant 6 bits, codes[-1] the least.
    Matches the inverse of decoder.py:_bitstring_to_codes.
    """
    if len(codes) != n_bonds:
        raise ValueError(
            f"codes length {len(codes)} != n_bonds {n_bonds}"
        )
    bs = 0
    mask = (1 << BITS_PER_BOND) - 1
    for i, c in enumerate(codes):
        ci = int(c)
        if ci < 0 or ci > mask:
            raise ValueError(
                f"code at position {i} = {ci} out of range [0,{mask}]"
            )
        shift = (n_bonds - 1 - i) * BITS_PER_BOND
        bs |= ci << shift
    return bs


def bitstring_int_to_codes(bitstring: int, n_bonds: int) -> np.ndarray:
    """Inverse of codes_to_bitstring_int. Returns shape (n_bonds,) int64."""
    mask = (1 << BITS_PER_BOND) - 1
    out = np.zeros(n_bonds, dtype=np.int64)
    for i in range(n_bonds):
        shift = (n_bonds - 1 - i) * BITS_PER_BOND
        out[i] = (bitstring >> shift) & mask
    return out


def codes_to_bitstring_str(codes: Sequence[int], n_bonds: int) -> str:
    """Pack codes into a binary string of length n_bonds*BITS_PER_BOND.

    No "0b" prefix. Length is exactly n_bonds*BITS_PER_BOND.
    """
    bs_int = codes_to_bitstring_int(codes, n_bonds)
    width = n_bonds * BITS_PER_BOND
    if width == 0:
        return ""
    return format(bs_int, f"0{width}b")


def bitstring_str_to_int(bitstring: str) -> int:
    """Parse a '0'/'1' string into an int. Empty string -> 0."""
    if bitstring is None:
        raise ValueError("bitstring is None")
    s = bitstring.strip()
    if s == "":
        return 0
    if s.startswith("0b") or s.startswith("0B"):
        s = s[2:]
    if not all(c in "01" for c in s):
        raise ValueError(f"non-binary char in bitstring: {bitstring!r}")
    return int(s, 2)


# ---------------------------------------------------------------------- #
# CandidateSample                                                        #
# ---------------------------------------------------------------------- #

@dataclass
class CandidateSample:
    """One decoded / scored candidate.

    At construction time at least one of (bitstring, codes) must be
    non-None. ``coords`` is None until decode_and_validate is run.

    Validity / scoring contract
    ---------------------------
    - invalid sample (valid=False): full_energy MUST stay None and the
      sample MUST NOT enter the refinement subspace.
    - accepted=False samples are kept in batches for statistics but must
      not enter full scoring or refinement.
    - filter_terms / full_energy_terms hold per-term contributions (after
      weighting) emitted by the corresponding scorer.
    - metadata is the catch-all for non-canonical fields (e.g. tau,
      acceptance random number, decode status).
    """
    bitstring: Optional[str] = None
    codes: Optional[List[int]] = None
    coords: Optional[np.ndarray] = None
    count: int = 1
    base_probability: Optional[float] = None
    filtered_probability: Optional[float] = None
    accepted: bool = False
    valid: bool = False
    invalid_reason: Optional[str] = None
    filter_energy: Optional[float] = None
    full_energy: Optional[float] = None
    filter_terms: Dict[str, float] = field(default_factory=dict)
    full_energy_terms: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.bitstring is None and self.codes is None:
            raise ValueError(
                "CandidateSample requires at least one of "
                "(bitstring, codes) to be non-None"
            )

    # -- convenience -----------------------------------------------------
    def as_bitstring_int(self, n_bonds: int) -> int:
        """Return bitstring as int, deriving from codes if needed."""
        if self.bitstring is not None:
            return bitstring_str_to_int(self.bitstring)
        return codes_to_bitstring_int(self.codes, n_bonds)

    def is_eligible_for_refinement(self) -> bool:
        return (
            self.accepted
            and self.valid
            and self.coords is not None
            and self.full_energy is not None
        )


# ---------------------------------------------------------------------- #
# SampleBatch                                                            #
# ---------------------------------------------------------------------- #

@dataclass
class SampleBatch:
    """A batch of CandidateSamples, typically for a single tau.

    Aggregate counters (n_raw, n_accepted, n_valid, ..., acceptance_rate,
    valid_rate, unique_bitstrings) are populated by the producer (the
    sampler). They are NOT recomputed on the fly by the methods below;
    the helper methods below act on the stored ``samples`` list.
    """
    samples: List[CandidateSample] = field(default_factory=list)
    tau: Optional[float] = None
    n_raw: int = 0
    n_accepted: int = 0
    n_valid: int = 0
    n_invalid: int = 0
    acceptance_rate: float = 0.0
    valid_rate: float = 0.0
    unique_bitstrings: int = 0
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- queries ---------------------------------------------------------
    def valid_samples(self) -> List[CandidateSample]:
        return [s for s in self.samples if s.valid]

    def accepted_samples(self) -> List[CandidateSample]:
        return [s for s in self.samples if s.accepted and s.valid]

    def top_by_filter_energy(self, k: int) -> List[CandidateSample]:
        cand = [
            s for s in self.samples
            if s.valid and s.filter_energy is not None
        ]
        cand.sort(key=lambda s: s.filter_energy)
        return cand[: max(0, int(k))]

    def top_by_full_energy(self, k: int) -> List[CandidateSample]:
        # Note: this method intentionally does NOT require coords to be
        # non-None — full_energy can come from any scorer that doesn't
        # need coords (or from an externally pre-scored sample). The
        # refinement subspace selection has its own stricter eligibility
        # gate (CandidateSample.is_eligible_for_refinement).
        cand = [
            s for s in self.samples
            if s.valid and s.accepted and s.full_energy is not None
        ]
        cand.sort(key=lambda s: s.full_energy)
        return cand[: max(0, int(k))]

    def report(self) -> Dict[str, Any]:
        """Plain dict of headline statistics. Safe to dump to JSON-ish logs."""
        return {
            "tau": self.tau,
            "n_raw": self.n_raw,
            "n_accepted": self.n_accepted,
            "n_valid": self.n_valid,
            "n_invalid": self.n_invalid,
            "acceptance_rate": self.acceptance_rate,
            "valid_rate": self.valid_rate,
            "unique_bitstrings": self.unique_bitstrings,
            "summary": dict(self.summary),
        }


__all__ = [
    "CandidateSample",
    "SampleBatch",
    "codes_to_bitstring_int",
    "bitstring_int_to_codes",
    "codes_to_bitstring_str",
    "bitstring_str_to_int",
]
