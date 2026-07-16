# Author: Yuqi Zhang
"""H_filter — diagonal, non-negative, low-cost filter Hamiltonian for the
imaginary-time-inspired classical-rejection sampler.

H_filter(z) = lambda_clash             * H_clash_proxy(z)
            + lambda_favorable_contact * H_favorable_contact_miss(z)
            + lambda_anchor            * H_anchor_tail(z)
            + lambda_bad_contact       * H_bad_contact_proxy(z)

Every term is non-negative by construction. The total is also
non-negative; if a numerical edge case produces a negative total, we
raise immediately rather than silently coerce.

H_favorable_contact_miss penalizes the absence of mid-range contacts
between residue pairs that the MJ table marks as attractive. It is
defined as

    s_ij = max(-W_MJ[i,j], 0)         (attraction strength, sign-aware)
    k(d) = exp(-(d / d_contact)^2)    (smooth contact kernel)
    H_fav = sum_{3 <= j-i <= contact_span} s_ij * (1 - k(d_ij))

This term replaces the role that H_bad_contact_proxy was supposed to
play with the canonical project MJ table — that table has no positive
entries, so max(W_MJ[i,j], 0) is identically 0 and provides no contact
guidance. H_bad_contact_proxy is retained for tables that DO carry
positive (unfavorable) entries; under the canonical table it
contributes 0, which is correct.

H_filter is intended ONLY for sampling-time filtering. It is NOT a
substitute for the full physical energy (FullEnergyScorer). It excludes:
  - global all-pair long-range clash (uses a window-bounded proxy only)
  - radius of gyration
  - signed full-pair contact energy (FullEnergyScorer.E_contact_full)
  - turn / dihedral preferences

See the project spec section 3.1 for the design rationale.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ras_folding.sampler.context import get_encoder_inputs, get_sequence
from ras_folding.scoring.geometry_terms import (
    clash_proxy_count,
    contact_indicator_pairs,
    direction_penalty,
    endpoint_distance_penalty,
    _gap_pair_indices,
)
from ras_folding.scoring.mj_contact import MJContactTable
from ras_folding.sampler.sample_types import CandidateSample


_DEFAULT_TERM_WEIGHTS: Dict[str, float] = {
    "clash_proxy": 5.0,
    "favorable_contact_miss": 0.3,
    "anchor_tail": 2.0,
    "bad_contact_proxy": 1.0,
}


class FilterHamiltonian:
    """Diagonal H_filter aggregator. Stateless w.r.t. samples."""

    def __init__(
        self,
        term_weights: Optional[Dict[str, float]] = None,
        clash_span: int = 5,
        contact_span: int = 5,
        anchor_tail_len: int = 3,
        d_min: float = 3.8,
        d_contact: float = 8.0,
        residue_contact_weights: Any = None,
        normalize_terms: bool = False,
        sigma_anchor: float = 3.8,
        anchor_cap: float = 10.0,
    ) -> None:
        if term_weights is None:
            term_weights = dict(_DEFAULT_TERM_WEIGHTS)
        else:
            merged = dict(_DEFAULT_TERM_WEIGHTS)
            merged.update(term_weights)
            term_weights = merged
        for k, v in term_weights.items():
            if v < 0:
                raise ValueError(
                    f"term weight {k}={v} is negative; H_filter must be >= 0"
                )
        self.term_weights = term_weights
        self.clash_span = int(clash_span)
        self.contact_span = int(contact_span)
        self.anchor_tail_len = int(anchor_tail_len)
        self.d_min = float(d_min)
        self.d_contact = float(d_contact)
        self.residue_contact_weights = residue_contact_weights
        self.normalize_terms = bool(normalize_terms)
        self.sigma_anchor = float(sigma_anchor)
        self.anchor_cap = float(anchor_cap)

        if self.clash_span < 3:
            raise ValueError(
                f"clash_span must be >= 3, got {self.clash_span}"
            )
        if self.contact_span < 3:
            raise ValueError(
                f"contact_span must be >= 3, got {self.contact_span}"
            )
        if self.d_min <= 0 or self.d_contact <= 0:
            raise ValueError("d_min and d_contact must be > 0")

    # ------------------------------------------------------------------ #
    # main entry                                                         #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        sample: CandidateSample,
        context_or_inputs,
    ) -> Tuple[float, Dict[str, float]]:
        """Compute H_filter(sample) and a per-term breakdown.

        Accepts either a SamplingContext or a raw EncoderInputs. Requires
        sample.coords to be set (decode_and_validate first). Returns
        (total, terms). `terms` holds WEIGHTED contributions
        (i.e. lambda_k * raw_k).

        Raises
        ------
        ValueError if sample.coords is None or H_filter < 0 (numerically).
        """
        if sample.coords is None:
            raise ValueError(
                "FilterHamiltonian.evaluate requires sample.coords; "
                "call decode_and_validate first."
            )
        encoder_inputs = get_encoder_inputs(context_or_inputs)
        sequence = get_sequence(context_or_inputs)
        coords = np.asarray(sample.coords, dtype=np.float64)

        terms: Dict[str, float] = {}
        raw: Dict[str, float] = {}

        # --- 1. clash proxy ---------------------------------------------
        h_clash, raw_clash, n_pairs_clash = self._h_clash_proxy(coords)
        raw["clash_proxy"] = float(raw_clash)
        terms["clash_proxy"] = self.term_weights["clash_proxy"] * h_clash

        # --- 2. favorable contact miss (MJ-aware, sign-aware) -----------
        h_fav, raw_fav, fav_meta = self._h_favorable_contact_miss(
            coords, sequence,
        )
        raw["favorable_contact_miss"] = float(raw_fav)
        terms["favorable_contact_miss"] = (
            self.term_weights["favorable_contact_miss"] * h_fav
        )

        # --- 3. anchor / tail -------------------------------------------
        h_anchor, anchor_meta = self._h_anchor_tail(coords, encoder_inputs)
        raw["anchor_tail"] = float(h_anchor)
        terms["anchor_tail"] = self.term_weights["anchor_tail"] * h_anchor

        # --- 4. bad contact proxy (kept; 0 under canonical MJ) ----------
        h_bad, raw_bad, bad_meta = self._h_bad_contact_proxy(
            coords, sequence,
        )
        raw["bad_contact_proxy"] = float(raw_bad)
        terms["bad_contact_proxy"] = (
            self.term_weights["bad_contact_proxy"] * h_bad
        )

        total = float(sum(terms.values()))
        if total < 0.0:
            raise ValueError(
                f"H_filter total is negative ({total}). "
                "All component terms must be >= 0."
            )

        # surface raw values + per-term metadata under sample.metadata
        meta = sample.metadata.setdefault("filter_hamiltonian", {})
        meta["raw"] = raw
        meta["clash_pair_window"] = n_pairs_clash
        meta["favorable_contact_miss"] = fav_meta
        meta["bad_contact"] = bad_meta
        meta["anchor"] = anchor_meta
        meta["weights"] = dict(self.term_weights)
        meta["normalize_terms"] = self.normalize_terms

        return total, terms

    # ------------------------------------------------------------------ #
    # individual terms                                                   #
    # ------------------------------------------------------------------ #

    def _h_clash_proxy(
        self, coords: np.ndarray,
    ) -> Tuple[float, int, int]:
        """Local clash proxy: count of i<j with 3<=j-i<=clash_span and
        ||r_i-r_j|| < d_min.

        Returns
        -------
        (h, raw_count, n_pairs_in_window)
            h is the value used in H_filter (raw or normalized).
        """
        n_clash, n_pairs = clash_proxy_count(
            coords, gap_min=3, gap_max=self.clash_span, d_min=self.d_min,
        )
        if self.normalize_terms and n_pairs > 0:
            return float(n_clash) / float(n_pairs), n_clash, n_pairs
        return float(n_clash), n_clash, n_pairs

    def _h_bad_contact_proxy(
        self,
        coords: np.ndarray,
        sequence,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """Window-bounded bad-contact proxy.

        Sum over i<j with 3 <= j-i <= contact_span of max(w_ij, 0) * q_ij,
        where q_ij = 1 if ||r_i-r_j|| < d_contact, else 0.

        If residue_contact_weights is None, an MJContactTable, or a
        pre-built (L,L) ndarray, we attempt to derive weights. If the
        table's sign convention is "unknown", or the sequence is missing,
        OR the (L,L) matrix has no positive entries (so max(w_ij,0)==0
        everywhere), we surface that fact and return 0 — never silently
        substituting a different score.
        """
        meta: Dict[str, Any] = {
            "mj_sign_unknown": False,
            "contact_weights_missing": False,
            "all_weights_nonpositive": False,
            "sequence_missing": sequence is None,
        }
        rcw = self.residue_contact_weights

        L = coords.shape[0]
        W: Optional[np.ndarray] = None
        if rcw is None:
            meta["contact_weights_missing"] = True
            return 0.0, 0.0, meta

        if isinstance(rcw, MJContactTable):
            if rcw.sign_convention == "unknown":
                meta["mj_sign_unknown"] = True
                return 0.0, 0.0, meta
            if sequence is None or len(sequence) != L:
                meta["contact_weights_missing"] = True
                meta["sequence_len"] = (
                    None if sequence is None else len(sequence)
                )
                meta["coords_len"] = L
                return 0.0, 0.0, meta
            W_seq, missing = rcw.weight_matrix_for_sequence(sequence)
            if missing:
                meta["missing_residues"] = missing
            if rcw.sign_convention == "negative_favorable":
                # bad = positive entries; canonical MJ has none.
                W = W_seq
            elif rcw.sign_convention == "positive_favorable":
                W = -W_seq
            else:
                meta["mj_sign_unknown"] = True
                return 0.0, 0.0, meta
        elif isinstance(rcw, np.ndarray):
            if rcw.shape != (L, L):
                meta["contact_weights_missing"] = True
                meta["bad_shape"] = tuple(rcw.shape)
                return 0.0, 0.0, meta
            W = rcw.astype(np.float64, copy=False)
        else:
            meta["contact_weights_missing"] = True
            meta["weight_type"] = type(rcw).__name__
            return 0.0, 0.0, meta

        I, J, q = contact_indicator_pairs(
            coords,
            gap_min=3,
            gap_max=self.contact_span,
            d_contact=self.d_contact,
        )
        if I.size == 0:
            meta["pair_window"] = 0
            return 0.0, 0.0, meta

        w_pairs = W[I, J]
        bad = np.maximum(w_pairs, 0.0) * q.astype(np.float64)
        if not bool(np.any(W > 0.0)):
            meta["all_weights_nonpositive"] = True
        raw = float(np.sum(bad))
        if self.normalize_terms and I.size > 0:
            return raw / float(I.size), raw, meta
        return raw, raw, meta

    def _h_favorable_contact_miss(
        self,
        coords: np.ndarray,
        sequence,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """H_favorable_contact_miss — penalize unrealised favorable contacts.

        For each window pair (i, j), 3 <= j-i <= contact_span:
            s_ij     = max(-W_MJ[i, j], 0)              (>= 0)
            kernel   = exp(-(d_ij / d_contact)^2)
            term_ij  = s_ij * (1 - kernel)              (>= 0)

        H_fav = sum term_ij. Non-negative by construction.
        """
        meta: Dict[str, Any] = {
            "mj_sign_unknown": False,
            "contact_weights_missing": False,
            "sequence_missing": sequence is None,
            "contact_span": self.contact_span,
            "d_contact": self.d_contact,
            "active_favorable_pairs": 0,
            "mj_sign_convention": None,
        }
        rcw = self.residue_contact_weights
        L = coords.shape[0]

        if rcw is None:
            meta["contact_weights_missing"] = True
            return 0.0, 0.0, meta

        # Resolve attraction matrix S (>= 0). Only enabled for sign-aware
        # tables / pre-built ndarrays.
        S: Optional[np.ndarray] = None
        if isinstance(rcw, MJContactTable):
            meta["mj_sign_convention"] = rcw.sign_convention
            if rcw.sign_convention == "unknown":
                meta["mj_sign_unknown"] = True
                return 0.0, 0.0, meta
            if sequence is None or len(sequence) != L:
                meta["contact_weights_missing"] = True
                meta["sequence_len"] = (
                    None if sequence is None else len(sequence)
                )
                meta["coords_len"] = L
                return 0.0, 0.0, meta
            S_seq, missing, available = rcw.attraction_matrix_for_sequence(
                sequence,
            )
            if not available:
                meta["mj_sign_unknown"] = True
                return 0.0, 0.0, meta
            if missing:
                meta["missing_residues"] = missing
            S = S_seq
        elif isinstance(rcw, np.ndarray):
            # Treat ndarray input as already-positive attraction matrix.
            if rcw.shape != (L, L):
                meta["contact_weights_missing"] = True
                meta["bad_shape"] = tuple(rcw.shape)
                return 0.0, 0.0, meta
            S = np.maximum(rcw.astype(np.float64, copy=False), 0.0)
        else:
            meta["contact_weights_missing"] = True
            meta["weight_type"] = type(rcw).__name__
            return 0.0, 0.0, meta

        I, J = _gap_pair_indices(L, gap_min=3, gap_max=self.contact_span)
        if I.size == 0:
            meta["pair_window"] = 0
            return 0.0, 0.0, meta

        s_pairs = S[I, J]
        # Distance per pair
        d = np.linalg.norm(coords[I] - coords[J], axis=1)
        # Smooth kernel; both d and d_contact > 0
        kernel = np.exp(-(d * d) / (self.d_contact * self.d_contact))
        contrib = s_pairs * (1.0 - kernel)
        # Numerical safety: any tiny FP negatives → clamp to 0
        contrib = np.maximum(contrib, 0.0)
        raw = float(np.sum(contrib))
        meta["active_favorable_pairs"] = int(np.count_nonzero(s_pairs > 0))
        meta["pair_window"] = int(I.size)
        if self.normalize_terms and I.size > 0:
            return raw / float(I.size), raw, meta
        return raw, raw, meta

    def _h_anchor_tail(
        self,
        coords: np.ndarray,
        encoder_inputs,
    ) -> Tuple[float, Dict[str, Any]]:
        """endpoint_penalty + direction_penalty (both >= 0).

        The penalty is the sum of:
          - endpoint_distance_penalty (capped quadratic)
          - 1 - cos(last_bond_dir, -v_right_seed) ∈ [0, 2]
        """
        anchor_right = getattr(encoder_inputs, "anchor_right", None)
        v_right_seed = getattr(encoder_inputs, "v_right_seed", None)

        ep, ep_avail = endpoint_distance_penalty(
            coords,
            anchor_right,
            sigma_anchor=self.sigma_anchor,
            cap=self.anchor_cap,
        )
        dp, dp_avail = direction_penalty(coords, v_right_seed)

        meta: Dict[str, Any] = {
            "endpoint_available": ep_avail,
            "direction_available": dp_avail,
            "endpoint_penalty_raw": ep,
            "direction_penalty_raw": dp,
        }
        return float(ep + dp), meta


__all__ = ["FilterHamiltonian"]
