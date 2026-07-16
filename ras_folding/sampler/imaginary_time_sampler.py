# Author: Yuqi Zhang
"""QuantumImaginaryTimeSampler — classical-rejection imaginary-time-inspired
sampler.

Core formula
------------
For each tau in `taus` and each raw candidate z:

    p_filtered(z; tau)  ∝  p_base(z) * exp(-tau * H_filter(z))

We implement this as classical rejection on top of an EncoderBaseSampler:

    A(z) = exp(-tau * H_filter(z))         (acceptance probability)
    accept iff U ~ U(0, 1) <= A(z)

When tau == 0, A(z) == 1 for every valid z, so all valid samples are
accepted (matches the spec).

Pipeline
--------
For each tau:
  1. base_sampler.sample(encoder_inputs, n=shots_per_tau, seed=seed_tau)
  2. for each raw sample s:
       decode_and_validate(s, encoder_inputs, decoder)
       if not s.valid:
           s.accepted=False; record invalid_reason
       else:
           E_filter, terms = filter_hamiltonian.evaluate(s, encoder_inputs)
           s.filter_energy = E_filter
           s.filter_terms  = terms
           A = exp(-tau * E_filter)
           u = rng.random()
           s.metadata["acceptance_random"] = u
           s.accepted = (u <= A)
  3. for each accepted+valid sample, optionally call scorer.evaluate
     to compute s.full_energy and s.full_energy_terms.
  4. compose SampleBatch with aggregate stats and return list[SampleBatch].

Determinism
-----------
The constructor seed seeds a PCG64 root generator. For each tau we
spawn a child generator deterministically (root.spawn(idx)). This makes
the per-tau output independent of the order of taus and reproducible
across calls with the same constructor seed.

Restrictions (per spec)
-----------------------
- No postselection_mode.
- No ancilla_filter.
- H_filter is for filtering ONLY; full_energy is the physical energy.
- Invalid samples never enter full scoring.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ras_folding.sampler.base_sampler import EncoderBaseSampler
from ras_folding.sampler.context import get_encoder_inputs
from ras_folding.sampler.filter_hamiltonian import FilterHamiltonian
from ras_folding.sampler.sample_types import CandidateSample, SampleBatch
from ras_folding.sampler.validity import decode_and_validate


class QuantumImaginaryTimeSampler:
    """Classical-rejection imaginary-time-inspired sampler.

    Parameters
    ----------
    taus : sequence of imaginary-time strengths (>= 0)
    shots_per_tau : raw samples drawn per tau
    base_sampler : EncoderBaseSampler producing raw candidates
    filter_hamiltonian : FilterHamiltonian for H_filter
    scorer : optional callable (sample, encoder_inputs) -> (full_E, terms)
             applied to accepted+valid samples. If None, full_energy stays None.
    seed : int | None — root random seed (independent from base_sampler)
    strict_validity : bool — if False, invalid samples may be accepted
                       (NOT recommended; defaults to True). Even when
                       False, invalid samples never enter full scoring.
    max_candidates_per_tau : optional cap on accepted samples per tau.
                             If exceeded, the lowest-H_filter accepted
                             samples are kept; the rest are flipped to
                             accepted=False with metadata["truncated"]=True.
    """

    def __init__(
        self,
        taus: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.4),
        shots_per_tau: int = 4096,
        base_sampler: Optional[EncoderBaseSampler] = None,
        filter_hamiltonian: Optional[FilterHamiltonian] = None,
        scorer: Any = None,
        seed: Optional[int] = None,
        strict_validity: bool = True,
        max_candidates_per_tau: Optional[int] = None,
    ) -> None:
        if shots_per_tau <= 0:
            raise ValueError(f"shots_per_tau must be > 0, got {shots_per_tau}")
        for t in taus:
            if t < 0:
                raise ValueError(f"tau must be >= 0, got {t}")
        self.taus: List[float] = [float(t) for t in taus]
        self.shots_per_tau = int(shots_per_tau)
        self.base_sampler = base_sampler or EncoderBaseSampler(
            n_samples=self.shots_per_tau, seed=seed,
        )
        self.filter_hamiltonian = filter_hamiltonian or FilterHamiltonian()
        self.scorer = scorer
        self.seed = seed
        self.strict_validity = bool(strict_validity)
        self.max_candidates_per_tau = (
            None if max_candidates_per_tau is None
            else int(max_candidates_per_tau)
        )

    # ------------------------------------------------------------------ #
    def sample(
        self,
        context_or_inputs,
        decoder: Optional[Callable] = None,
    ) -> List[SampleBatch]:
        root = np.random.default_rng(self.seed)
        # Derive deterministic child seeds for each tau
        child_seeds = root.integers(
            low=0, high=np.iinfo(np.int64).max,
            size=len(self.taus), dtype=np.int64,
        ).tolist()

        batches: List[SampleBatch] = []
        for idx, tau in enumerate(self.taus):
            sub_seed = int(child_seeds[idx])
            batch = self._sample_one_tau(
                context_or_inputs, tau, sub_seed, decoder,
            )
            batches.append(batch)
        return batches

    # ------------------------------------------------------------------ #
    def _sample_one_tau(
        self,
        context_or_inputs,
        tau: float,
        sub_seed: int,
        decoder: Optional[Callable],
    ) -> SampleBatch:
        rng = np.random.default_rng(sub_seed)
        # Independent base-sampler draw: pass the same sub_seed so the
        # raw bitstrings are deterministic for this tau.
        raw = self.base_sampler.sample(
            context_or_inputs,
            n_samples=self.shots_per_tau,
            seed=sub_seed,
        )

        # Pre-draw acceptance uniforms for vectorized determinism.
        acc_u = rng.random(size=len(raw))

        n_valid = 0
        n_invalid = 0
        n_accepted = 0
        seen_bitstrings: Dict[str, int] = {}

        for i, s in enumerate(raw):
            s.metadata["tau"] = tau
            s.metadata["acceptance_random"] = float(acc_u[i])

            # 1) decode + validate
            decode_and_validate(s, context_or_inputs, decoder=decoder)

            if not s.valid:
                n_invalid += 1
                s.accepted = False
                # filter_energy stays None for invalid samples
                continue
            n_valid += 1

            # 2) compute H_filter
            try:
                e_filter, terms = self.filter_hamiltonian.evaluate(
                    s, context_or_inputs,
                )
            except Exception as e:
                # Hard surface: do not silently swallow. Mark invalid.
                s.valid = False
                s.invalid_reason = "unknown_invalid"
                s.metadata["filter_eval_exception"] = repr(e)
                s.accepted = False
                n_valid -= 1
                n_invalid += 1
                continue

            s.filter_energy = float(e_filter)
            s.filter_terms = dict(terms)

            # 3) acceptance test
            if tau == 0.0:
                A = 1.0
            else:
                # exp(-tau * E_filter) — clip exponent for numerical safety
                A = math.exp(-min(tau * e_filter, 700.0))
            s.base_probability = 1.0 / float(self.shots_per_tau)
            s.filtered_probability = float(A)
            s.accepted = bool(acc_u[i] <= A)
            if s.accepted:
                n_accepted += 1

            # 4) optional full scoring on accepted valid samples
            if s.accepted and self.scorer is not None:
                try:
                    e_full, full_terms = self.scorer.evaluate(
                        s, context_or_inputs,
                    )
                    s.full_energy = float(e_full)
                    s.full_energy_terms = dict(full_terms)
                except Exception as e:
                    # Keep accepted but record that full scoring failed.
                    s.full_energy = None
                    s.full_energy_terms = {}
                    s.metadata["full_score_exception"] = repr(e)

            if s.bitstring is not None:
                seen_bitstrings[s.bitstring] = (
                    seen_bitstrings.get(s.bitstring, 0) + 1
                )

        # 5) optional truncation by H_filter (keep lowest)
        truncated_count = 0
        if (
            self.max_candidates_per_tau is not None
            and n_accepted > self.max_candidates_per_tau
        ):
            accepted = [
                s for s in raw if s.accepted and s.valid
            ]
            accepted.sort(
                key=lambda s: (
                    float("inf") if s.filter_energy is None
                    else s.filter_energy
                )
            )
            keep = set(id(s) for s in accepted[: self.max_candidates_per_tau])
            for s in raw:
                if s.accepted and s.valid and id(s) not in keep:
                    s.accepted = False
                    s.metadata["truncated"] = True
                    truncated_count += 1
            n_accepted -= truncated_count

        # 6) batch assembly
        n_raw = len(raw)
        acc_rate = float(n_accepted) / float(n_raw) if n_raw else 0.0
        valid_rate = float(n_valid) / float(n_raw) if n_raw else 0.0
        unique = len(seen_bitstrings)

        invalid_reason_counts: Dict[str, int] = {}
        for s in raw:
            if not s.valid and s.invalid_reason:
                invalid_reason_counts[s.invalid_reason] = (
                    invalid_reason_counts.get(s.invalid_reason, 0) + 1
                )

        # --- diagnostics: H_filter / E_full means and correlations ----
        diag = _filter_full_diagnostics(raw)

        summary = {
            "invalid_reason_counts": invalid_reason_counts,
            "shots": n_raw,
            "tau": tau,
            "truncated": truncated_count,
            "scorer_attached": self.scorer is not None,
            "filter_weights": dict(
                self.filter_hamiltonian.term_weights
            ),
            **diag,
        }

        return SampleBatch(
            samples=raw,
            tau=tau,
            n_raw=n_raw,
            n_accepted=n_accepted,
            n_valid=n_valid,
            n_invalid=n_invalid,
            acceptance_rate=acc_rate,
            valid_rate=valid_rate,
            unique_bitstrings=unique,
            summary=summary,
            metadata={
                "sub_seed": sub_seed,
                "strict_validity": self.strict_validity,
            },
        )


# ---------------------------------------------------------------------- #
# diagnostics                                                            #
# ---------------------------------------------------------------------- #

def _pearson(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size < 2 or y.size < 2:
        return None
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std < 1e-12 or y_std < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size < 2 or y.size < 2:
        return None
    # If the raw values are constant, ranks are meaningless — return None.
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return None
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    if float(np.std(rx)) < 1e-12 or float(np.std(ry)) < 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _filter_full_diagnostics(samples) -> Dict[str, Any]:
    """Return mean/best/correlation diagnostics computed only over the
    valid (and where applicable, accepted) subset."""
    full_valid = [
        s.full_energy for s in samples
        if s.valid and s.full_energy is not None
    ]
    full_accepted = [
        s.full_energy for s in samples
        if s.valid and s.accepted and s.full_energy is not None
    ]

    out: Dict[str, Any] = {
        "mean_full_energy_valid": (
            float(np.mean(full_valid)) if full_valid else None
        ),
        "mean_full_energy_accepted": (
            float(np.mean(full_accepted)) if full_accepted else None
        ),
        "best_full_energy_accepted": (
            float(np.min(full_accepted)) if full_accepted else None
        ),
        "top10_mean_full_energy_accepted": None,
        "mean_filter_energy_valid": None,
        "mean_filter_energy_accepted": None,
        "pearson_corr_filter_full": None,
        "spearman_corr_filter_full": None,
        "n_filter_full_corr_samples": 0,
    }

    if full_accepted:
        sorted_full = sorted(full_accepted)
        k = min(10, len(sorted_full))
        out["top10_mean_full_energy_accepted"] = float(
            np.mean(sorted_full[:k])
        )

    f_valid = [
        s.filter_energy for s in samples
        if s.valid and s.filter_energy is not None
    ]
    if f_valid:
        out["mean_filter_energy_valid"] = float(np.mean(f_valid))
    f_accepted = [
        s.filter_energy for s in samples
        if s.valid and s.accepted and s.filter_energy is not None
    ]
    if f_accepted:
        out["mean_filter_energy_accepted"] = float(np.mean(f_accepted))

    paired = [
        (s.filter_energy, s.full_energy)
        for s in samples
        if s.valid and s.filter_energy is not None
        and s.full_energy is not None
    ]
    out["n_filter_full_corr_samples"] = len(paired)
    if len(paired) >= 2:
        x = np.array([p[0] for p in paired], dtype=np.float64)
        y = np.array([p[1] for p in paired], dtype=np.float64)
        out["pearson_corr_filter_full"] = _pearson(x, y)
        out["spearman_corr_filter_full"] = _spearman(x, y)

    return out


__all__ = ["QuantumImaginaryTimeSampler"]
