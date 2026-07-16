# Author: Yuqi Zhang
"""FullEnergyScorer — full physical energy applied AFTER validity / acceptance.

E_full(z) = lambda_overlap * E_overlap_full(z)
          + lambda_contact * E_contact_full(z)
          + lambda_rg      * E_rg(z)
          + lambda_anchor  * E_anchor(z)
          + lambda_turn    * E_turn(z)

Conventions
-----------
- Lower energy is better.
- Only valid + accepted samples should be passed in (the caller
  enforces this — Validity wrapper / imaginary-time sampler).
- E_contact_full uses the MJ table directly (signed). With the standard
  "negative_favorable" sign, favorable contacts contribute negative
  energy.
- E_overlap_full is the strict global clash count (all pairs |i-j|>=3).
- E_rg is zero unless rg_target is supplied — we never fabricate a target.
- E_anchor is zero when anchor_right is unavailable.
- E_turn is zero by default (the encoder lattice already enforces local
  bend windows). Implementation included as a tail-bend deviation
  penalty if the caller enables it.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ras_folding.encoder.lattice import BEND_MAX_DEG, BEND_MIN_DEG
from ras_folding.sampler.context import get_encoder_inputs, get_sequence
from ras_folding.scoring.geometry_terms import (
    clash_full_count,
    contact_indicator_pairs,
    direction_penalty,
    endpoint_distance_penalty,
    radius_of_gyration,
    tail_bend_penalty,
)
from ras_folding.scoring.mj_contact import MJContactTable
from ras_folding.sampler.sample_types import CandidateSample


_DEFAULT_TERM_WEIGHTS: Dict[str, float] = {
    "overlap_full": 10.0,
    "contact_full": 1.0,
    "rg": 1.0,
    "anchor": 1.0,
    "turn": 0.0,  # off by default; encoder lattice enforces local bend
}


class FullEnergyScorer:
    """Aggregator for the full physical energy.

    Each call to ``evaluate`` returns (E_full, terms) and merges raw
    diagnostics into sample.metadata under "full_energy".
    """

    def __init__(
        self,
        term_weights: Optional[Dict[str, float]] = None,
        d_min: float = 3.8,
        d_contact: float = 8.0,
        residue_contact_weights: Any = None,
        rg_target: Optional[float] = None,
        sigma_rg: float = 1.0,
        sigma_anchor: float = 3.8,
        anchor_cap: float = 1e6,
        turn_tail_len: int = 0,
    ) -> None:
        if term_weights is None:
            term_weights = dict(_DEFAULT_TERM_WEIGHTS)
        else:
            merged = dict(_DEFAULT_TERM_WEIGHTS)
            merged.update(term_weights)
            term_weights = merged
        self.term_weights = term_weights
        self.d_min = float(d_min)
        self.d_contact = float(d_contact)
        self.residue_contact_weights = residue_contact_weights
        self.rg_target = (None if rg_target is None else float(rg_target))
        self.sigma_rg = float(sigma_rg)
        self.sigma_anchor = float(sigma_anchor)
        self.anchor_cap = float(anchor_cap)
        self.turn_tail_len = int(turn_tail_len)

        if self.d_min <= 0 or self.d_contact <= 0 or self.sigma_rg <= 0 or self.sigma_anchor <= 0:
            raise ValueError("d_min, d_contact, sigma_rg, sigma_anchor must be > 0")

    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        sample: CandidateSample,
        context_or_inputs,
    ) -> Tuple[float, Dict[str, float]]:
        if sample.coords is None:
            raise ValueError(
                "FullEnergyScorer.evaluate requires sample.coords; "
                "call decode_and_validate first."
            )
        if not sample.valid:
            raise ValueError(
                "FullEnergyScorer.evaluate must not be called on invalid "
                "samples (sample.valid is False)."
            )
        encoder_inputs = get_encoder_inputs(context_or_inputs)
        sequence = get_sequence(context_or_inputs)
        coords = np.asarray(sample.coords, dtype=np.float64)
        meta: Dict[str, Any] = {
            "rg_target_missing": self.rg_target is None,
            "anchor_unavailable": False,
            "contact_unavailable": False,
            "mj_sign_unknown": False,
            "sequence_missing": sequence is None,
        }

        terms_weighted: Dict[str, float] = {}

        # 1. global overlap ------------------------------------------------
        n_clash, n_pairs_clash = clash_full_count(
            coords, gap_min=3, d_min=self.d_min,
        )
        e_overlap = float(n_clash)
        terms_weighted["overlap_full"] = (
            self.term_weights["overlap_full"] * e_overlap
        )
        meta["overlap_full_raw"] = n_clash
        meta["overlap_full_pairs"] = n_pairs_clash

        # 2. full contact (MJ-weighted) ----------------------------------
        e_contact, contact_meta = self._e_contact_full(coords, sequence)
        terms_weighted["contact_full"] = (
            self.term_weights["contact_full"] * e_contact
        )
        meta["contact_full_raw"] = e_contact
        meta.update(contact_meta)

        # 3. Rg ----------------------------------------------------------
        rg = radius_of_gyration(coords)
        meta["rg"] = rg
        if self.rg_target is None:
            e_rg = 0.0
        else:
            dev = (rg - self.rg_target) / self.sigma_rg
            e_rg = float(dev * dev)
        terms_weighted["rg"] = self.term_weights["rg"] * e_rg
        meta["rg_raw"] = e_rg

        # 4. anchor -------------------------------------------------------
        anchor_right = getattr(encoder_inputs, "anchor_right", None)
        v_right_seed = getattr(encoder_inputs, "v_right_seed", None)
        ep, ep_avail = endpoint_distance_penalty(
            coords,
            anchor_right,
            sigma_anchor=self.sigma_anchor,
            cap=self.anchor_cap,
        )
        dp, dp_avail = direction_penalty(coords, v_right_seed)
        if not ep_avail and not dp_avail:
            meta["anchor_unavailable"] = True
        e_anchor = float(ep + dp)
        terms_weighted["anchor"] = self.term_weights["anchor"] * e_anchor
        meta["anchor_endpoint_raw"] = ep
        meta["anchor_direction_raw"] = dp
        meta["anchor_endpoint_available"] = ep_avail
        meta["anchor_direction_available"] = dp_avail

        # 5. turn (optional) ---------------------------------------------
        if self.term_weights.get("turn", 0.0) != 0.0 and self.turn_tail_len > 0:
            e_turn = tail_bend_penalty(
                coords,
                tail_len=self.turn_tail_len,
                bend_min_deg=BEND_MIN_DEG,
                bend_max_deg=BEND_MAX_DEG,
            )
        else:
            e_turn = 0.0
        terms_weighted["turn"] = self.term_weights.get("turn", 0.0) * e_turn
        meta["turn_raw"] = e_turn

        total = float(sum(terms_weighted.values()))
        sample.metadata.setdefault("full_energy", {}).update(meta)
        sample.metadata["full_energy"]["weights"] = dict(self.term_weights)
        return total, terms_weighted

    # ------------------------------------------------------------------ #
    def _e_contact_full(
        self,
        coords: np.ndarray,
        sequence,
    ) -> Tuple[float, Dict[str, Any]]:
        meta: Dict[str, Any] = {}
        rcw = self.residue_contact_weights
        L = coords.shape[0]

        if rcw is None:
            meta["contact_unavailable"] = True
            meta["contact_reason"] = "no_residue_contact_weights"
            return 0.0, meta

        if isinstance(rcw, MJContactTable):
            if rcw.sign_convention == "unknown":
                meta["mj_sign_unknown"] = True
                meta["contact_unavailable"] = True
                return 0.0, meta
            if sequence is None or len(sequence) != L:
                meta["contact_unavailable"] = True
                meta["contact_reason"] = "sequence_missing_or_mismatch"
                meta["sequence_len"] = (
                    None if sequence is None else len(sequence)
                )
                meta["coords_len"] = L
                return 0.0, meta
            W_seq, missing = rcw.weight_matrix_for_sequence(sequence)
            if missing:
                meta["missing_residues"] = list(missing)
            if rcw.sign_convention == "negative_favorable":
                W = W_seq          # negative = favorable -> lower E
            else:  # positive_favorable
                W = -W_seq         # invert so negative remains "favorable"
        elif isinstance(rcw, np.ndarray):
            if rcw.shape != (L, L):
                meta["contact_unavailable"] = True
                meta["contact_reason"] = "ndarray_bad_shape"
                meta["bad_shape"] = tuple(rcw.shape)
                return 0.0, meta
            W = rcw.astype(np.float64, copy=False)
        else:
            meta["contact_unavailable"] = True
            meta["contact_reason"] = f"unsupported_type_{type(rcw).__name__}"
            return 0.0, meta

        # full contact: all pairs with j-i >= 3
        I, J, q = contact_indicator_pairs(
            coords,
            gap_min=3,
            gap_max=None,
            d_contact=self.d_contact,
        )
        if I.size == 0:
            return 0.0, meta
        w = W[I, J] * q.astype(np.float64)
        return float(np.sum(w)), meta


__all__ = ["FullEnergyScorer"]
