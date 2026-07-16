# Author: Yuqi Zhang
"""Validity wrapper around the encoder's decoder.

The encoder's ``decode_bitstring`` returns a coordinate array regardless
of whether the autoregressive walk encountered the fallback branch
(decoder.py:104-122). It exposes no validity flag. This module wraps the
decode + post-decode validity check into a single entry point that
populates ``CandidateSample.valid`` and ``CandidateSample.invalid_reason``.

Public API
----------
decode_and_validate(sample, encoder_inputs, decoder=None) -> CandidateSample
validate_decoded_coords(coords, encoder_inputs) -> (valid, reason, info)

Invalid-reason vocabulary (matching the spec)
---------------------------------------------
  decode_failed            : decode raised, or returned None / wrong type
  nan_coords               : NaN/inf in the coordinate array
  bad_shape                : shape != (n_residues, 3)
  bond_length_violation    : a CA-CA bond length deviates from CA_CA_LENGTH
                              by more than BOND_TOL_A
  long_range_clash         : ||r_i - r_j|| < MIN_SEP for some |i-j| >= 3
  endpoint_mismatch        : ||r_last - anchor_right|| > epsilon_endpoint
  direction_mismatch       : final bond direction inconsistent with
                              -v_right_seed beyond cosine tolerance
  fallback_triggered       : decoder explicitly reported fallback
                              (only if the caller passes such metadata
                              via sample.metadata["fallback_triggered"];
                              the stock decoder.py:60 does not emit it)
  unknown_invalid          : catch-all for unexpected exceptions

Tolerances
----------
BOND_TOL_A         = 1e-6 (decoder writes CA_CA_LENGTH * unit_vec exactly,
                      so any meaningful deviation is a real bug)
EPSILON_ENDPOINT_A = ras_folding.encoder.decoder.EPSILON (1.0)
DIR_COS_TOL        = cos(BEND_MAX_DEG)
                     (i.e. the same window the encoder enforces at the
                      last bond when remaining=0 in reachable.py:195-198)
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ras_folding.encoder.decoder import (
    EPSILON,
    decode_bitstring,
    decode_bitstring_with_info,
)
from ras_folding.encoder.lattice import BEND_MAX_DEG, BEND_MIN_DEG
from ras_folding.encoder.reachable import MIN_SEP, SKIP_RECENT
from ras_folding.sampler.context import get_encoder_inputs
from ras_folding.sampler.sample_types import (
    CandidateSample,
    bitstring_str_to_int,
    codes_to_bitstring_int,
)
from ras_folding.utils.constants import CA_CA_LENGTH


# Tight bond-length tolerance: the decoder produces CA_CA_LENGTH * d_unit
# at every step, so deviations beyond ~1e-6 indicate a real defect.
BOND_TOL_A: float = 1e-6
EPSILON_ENDPOINT_A: float = float(EPSILON)
DIR_COS_TOL_HIGH: float = math.cos(math.radians(BEND_MIN_DEG))
DIR_COS_TOL_LOW: float = math.cos(math.radians(BEND_MAX_DEG))


# ---------------------------------------------------------------------- #
# Pure coord-level validation                                            #
# ---------------------------------------------------------------------- #

def validate_decoded_coords(
    coords: Optional[np.ndarray],
    context_or_inputs,
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Run the post-decode validity battery on a coordinate array.

    Returns
    -------
    (valid, reason, info)
        valid : bool — True only if every active check passes.
        reason : str | None — first invalid_reason encountered, else None.
        info : dict — diagnostic numbers (endpoint distance, max bond
               deviation, min long-range distance, etc.)
    """
    info: Dict[str, Any] = {}

    if coords is None:
        return False, "decode_failed", info

    if not isinstance(coords, np.ndarray):
        info["coords_type"] = type(coords).__name__
        return False, "bad_shape", info

    encoder_inputs = get_encoder_inputs(context_or_inputs)
    n = int(getattr(encoder_inputs, "n_residues", 0))
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] != n:
        info["coords_shape"] = tuple(coords.shape)
        info["expected_shape"] = (n, 3)
        return False, "bad_shape", info

    if not np.all(np.isfinite(coords)):
        info["nonfinite_count"] = int(np.sum(~np.isfinite(coords)))
        return False, "nan_coords", info

    # --- bond length check --------------------------------------------
    if n >= 2:
        bonds = coords[1:] - coords[:-1]
        bond_lens = np.linalg.norm(bonds, axis=1)
        max_dev = float(np.max(np.abs(bond_lens - CA_CA_LENGTH)))
        info["max_bond_dev"] = max_dev
        info["bond_len_min"] = float(bond_lens.min())
        info["bond_len_max"] = float(bond_lens.max())
        if max_dev > BOND_TOL_A:
            return False, "bond_length_violation", info
    else:
        info["max_bond_dev"] = 0.0

    # --- long-range clash ---------------------------------------------
    # SKIP_RECENT in reachable.py is per-step; here we want the *trace*
    # invariant, i.e. for any pair |i-j| >= 3.
    if n >= 4:
        diffs = coords[:, None, :] - coords[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        iu = np.arange(n)[:, None]
        ju = np.arange(n)[None, :]
        gap_mask = (ju - iu) >= 3
        if gap_mask.any():
            lr_min = float(dists[gap_mask].min())
            info["long_range_min_dist"] = lr_min
            if lr_min < MIN_SEP:
                return False, "long_range_clash", info
        else:
            info["long_range_min_dist"] = float("inf")
    else:
        info["long_range_min_dist"] = float("inf")

    # --- endpoint match -----------------------------------------------
    anchor_right = getattr(encoder_inputs, "anchor_right", None)
    if anchor_right is not None and n >= 1:
        ar = np.asarray(anchor_right, dtype=np.float64)
        endpoint_dist = float(np.linalg.norm(coords[-1] - ar))
        info["endpoint_dist"] = endpoint_dist
        if endpoint_dist > EPSILON_ENDPOINT_A:
            return False, "endpoint_mismatch", info
    else:
        info["endpoint_dist"] = None

    # --- direction match with v_right_seed ----------------------------
    v_right_seed = getattr(encoder_inputs, "v_right_seed", None)
    if v_right_seed is not None and n >= 2:
        last = coords[-1] - coords[-2]
        last_norm = float(np.linalg.norm(last))
        if last_norm < 1e-12:
            info["last_bond_zero"] = True
            return False, "direction_mismatch", info
        last_unit = last / last_norm
        v_target = -np.asarray(v_right_seed, dtype=np.float64)
        v_norm = float(np.linalg.norm(v_target))
        if v_norm < 1e-12:
            info["v_right_seed_zero"] = True
            # treat as unavailable rather than failing (user upstream may
            # have passed a degenerate seed); record but pass.
            info["direction_check"] = "skipped_zero_seed"
        else:
            v_target = v_target / v_norm
            cos_end = float(np.clip(np.dot(last_unit, v_target), -1.0, 1.0))
            ang_end = math.degrees(math.acos(cos_end))
            info["last_bond_vs_right_angle_deg"] = ang_end
            in_window = (
                cos_end >= DIR_COS_TOL_LOW and cos_end <= DIR_COS_TOL_HIGH
            )
            if not in_window:
                info["direction_window_deg"] = (
                    BEND_MIN_DEG, BEND_MAX_DEG,
                )
                return False, "direction_mismatch", info

    return True, None, info


# ---------------------------------------------------------------------- #
# decode + validate                                                      #
# ---------------------------------------------------------------------- #

def decode_and_validate(
    sample: CandidateSample,
    context_or_inputs,
    decoder: Optional[Callable] = None,
) -> CandidateSample:
    """Decode (if needed) and run validity battery on a sample.

    Parameters
    ----------
    sample : CandidateSample
        codes or bitstring must be set. coords may be None.
    context_or_inputs : SamplingContext or EncoderInputs
    decoder : optional callable or module
        Resolution order:
          1. If `decoder` is None and the sample has no coords, we call
             ``decode_bitstring_with_info(bs_int, inputs)`` and consume
             its info dict (canonical instrumented path).
          2. If `decoder` is a module/object exposing
             ``decode_bitstring_with_info``, we call that.
          3. Otherwise we call `decoder(bs_int, inputs)` and accept
             either a plain ndarray or a (coords, info) tuple.

    Behavior
    --------
    - Never raises on a malformed sample; populates valid/invalid_reason
      and returns the same CandidateSample object (mutated in place).
    - If the decoder raises, sample.valid=False, invalid_reason="decode_failed",
      metadata["decode_exception"] = repr(exc).
    - If info["fallback_triggered"] is True (from the instrumented
      decoder OR explicitly set on sample.metadata), the sample is
      marked invalid with reason="fallback_triggered" without further
      coord-level checks.
    """
    encoder_inputs = get_encoder_inputs(context_or_inputs)
    n_bonds = int(getattr(encoder_inputs, "n_bonds", 0))

    # --- decode (if coords missing) ------------------------------------
    if sample.coords is None:
        try:
            bs_int = _resolve_bitstring_int(sample, n_bonds)
        except Exception as e:
            sample.valid = False
            sample.invalid_reason = "decode_failed"
            sample.metadata["decode_exception"] = repr(e)
            return sample

        try:
            decoded, decode_info = _invoke_decoder(
                decoder, bs_int, encoder_inputs,
            )
        except Exception as e:
            sample.valid = False
            sample.invalid_reason = "decode_failed"
            sample.metadata["decode_exception"] = repr(e)
            return sample

        if isinstance(decode_info, dict):
            # surface canonical fields
            for k in (
                "fallback_triggered",
                "fallback_steps",
                "n_fallback_steps",
                "endpoint_residual",
            ):
                if k in decode_info:
                    sample.metadata[f"decode_{k}"] = decode_info[k]
            if decode_info.get("fallback_triggered"):
                sample.metadata["fallback_triggered"] = True

        sample.coords = decoded

    # --- explicit fallback signal ---------------------------------------
    if sample.metadata.get("fallback_triggered"):
        sample.valid = False
        sample.invalid_reason = "fallback_triggered"
        return sample

    # --- coord-level battery -------------------------------------------
    try:
        valid, reason, info = validate_decoded_coords(
            sample.coords, encoder_inputs,
        )
    except Exception as e:
        sample.valid = False
        sample.invalid_reason = "unknown_invalid"
        sample.metadata["validate_exception"] = repr(e)
        return sample

    sample.valid = bool(valid)
    sample.invalid_reason = reason
    sample.metadata["validity_info"] = info
    return sample


def _invoke_decoder(decoder, bs_int, encoder_inputs):
    """Resolve the decoder call and return (coords, info_or_None).

    Resolution order (matches the docstring of decode_and_validate):
      - decoder is None → decode_bitstring_with_info
      - decoder has attribute decode_bitstring_with_info → call it
      - decoder is a callable → call(bs_int, inputs); accept ndarray
        or (coords, info) tuple
    """
    if decoder is None:
        return decode_bitstring_with_info(bs_int, encoder_inputs)

    with_info = getattr(decoder, "decode_bitstring_with_info", None)
    if callable(with_info):
        return with_info(bs_int, encoder_inputs)

    if not callable(decoder):
        raise TypeError(
            f"decoder must be None, a callable, or expose "
            f"decode_bitstring_with_info; got {type(decoder).__name__}"
        )

    out = decoder(bs_int, encoder_inputs)
    if isinstance(out, tuple) and len(out) == 2:
        coords, info = out
        if not isinstance(info, dict):
            info = None
        return coords, info
    return out, None


def _resolve_bitstring_int(sample: CandidateSample, n_bonds: int) -> int:
    """Pull a bitstring int out of the sample's bitstring/codes."""
    if sample.bitstring is not None:
        return bitstring_str_to_int(sample.bitstring)
    if sample.codes is None:
        raise ValueError("sample has neither bitstring nor codes")
    return codes_to_bitstring_int(sample.codes, n_bonds)


__all__ = [
    "decode_and_validate",
    "validate_decoded_coords",
    "BOND_TOL_A",
    "EPSILON_ENDPOINT_A",
]
