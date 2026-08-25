# Author: Yuqi Zhang
"""PriorArmBaseSampler (arm P) — direct classical draws from the prior.

Purpose
-------
Arm P is the control that isolates the contribution of hardware-executed
circuit sampling. The quantum arm does not sample the prior directly: the
prior is compressed into HEA parameters by
``ras_folding.quantum.moment_match_initializer``, which retains only

  * the 1-point bit marginals ``p_q``, and
  * the 2-point correlations ``C_{a,b}`` on the CX edges of the ansatz,

and discards every higher-order and non-edge dependency of the
autoregressive joint. The circuit is therefore a lossy second-order
re-encoding of the same prior. Arm P draws from that prior exactly, under
the identical downstream, so the difference between the two arms is
attributable to the circuit/hardware step rather than to the prior.

Budget
------
The arms are not budget-comparable by shot count alone, because a prior
rollout that hits an infeasible step or misses the endpoint tolerance
yields no bitstring at all, whereas every quantum shot yields one. Both
matched definitions required by the revision plan are therefore supported
explicitly and both counts are always recorded:

``BudgetMode.PROPOSAL``
    Draw exactly ``n_proposals`` rollout attempts. Matches the quantum arm
    on proposals offered to the downstream (attempts <-> shots).

``BudgetMode.VALID``
    Keep drawing until ``n_valid_target`` valid paths are obtained, or the
    attempt cap is reached. Matches the quantum arm on valid candidates
    delivered to the downstream.

Never silently truncate: if a VALID-mode run exhausts its attempt cap
before reaching the target, the shortfall is recorded in the result
metadata and emitted through the logger, not hidden.

Contract
--------
Returns ``CandidateSample`` objects with ``coords=None``, ``valid=False``,
``accepted=False``. Coordinates are deliberately NOT forwarded from the
prior sampler even though it computes them internally: the downstream must
run its own ``decode_and_validate`` for every arm, or the arms would not
share a decoder path.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ras_folding.prior.prior_sampler import PriorConditionedBaseSampler
from ras_folding.sampler.context import get_encoder_inputs
from ras_folding.sampler.sample_types import CandidateSample

logger = logging.getLogger(__name__)

# Rollout attempts per internal chunk. Bounds peak memory and gives the
# VALID mode a granular stopping check without per-sample overhead.
DEFAULT_CHUNK_SIZE = 16_384

# Safety factor for VALID mode: cap attempts at
#   ceil(n_valid_target / observed_valid_rate) * ATTEMPT_CAP_SLACK
# once a rate has been observed, so a collapsing valid rate cannot spin.
ATTEMPT_CAP_SLACK = 3.0


class BudgetMode(str, enum.Enum):
    PROPOSAL = "proposal"
    VALID = "valid"


@dataclass
class PriorArmBudget:
    """Budget specification for one arm-P invocation."""
    mode: BudgetMode = BudgetMode.PROPOSAL
    n_proposals: Optional[int] = None      # required for PROPOSAL
    n_valid_target: Optional[int] = None   # required for VALID
    max_attempts: Optional[int] = None     # hard cap for VALID
    chunk_size: int = DEFAULT_CHUNK_SIZE

    def __post_init__(self) -> None:
        self.mode = BudgetMode(self.mode)
        if self.mode is BudgetMode.PROPOSAL:
            if not self.n_proposals or self.n_proposals <= 0:
                raise ValueError(
                    "PROPOSAL budget requires n_proposals > 0; "
                    f"got {self.n_proposals!r}"
                )
        else:
            if not self.n_valid_target or self.n_valid_target <= 0:
                raise ValueError(
                    "VALID budget requires n_valid_target > 0; "
                    f"got {self.n_valid_target!r}"
                )
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0; got {self.chunk_size}")


@dataclass
class PriorArmStats:
    """Honest accounting for one arm-P invocation."""
    n_attempts: int = 0
    n_valid: int = 0
    n_unique: int = 0
    n_chunks: int = 0
    valid_rate: float = 0.0
    budget_mode: str = ""
    target: Optional[int] = None
    target_met: bool = True
    shortfall: int = 0
    invalid_reason_counts: Dict[str, int] = field(default_factory=dict)
    chunk_seeds: List[int] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_attempts": int(self.n_attempts),
            "n_valid": int(self.n_valid),
            "n_unique": int(self.n_unique),
            "n_chunks": int(self.n_chunks),
            "valid_rate": float(self.valid_rate),
            "budget_mode": self.budget_mode,
            "target": self.target,
            "target_met": bool(self.target_met),
            "shortfall": int(self.shortfall),
            "invalid_reason_counts": dict(self.invalid_reason_counts),
            "chunk_seeds": [int(s) for s in self.chunk_seeds],
        }


class PriorArmBaseSampler:
    """Arm P. Same contract as ``QuantumBackendBaseSampler.sample``."""

    ARM = "P"

    def __init__(
        self,
        env_ctx: Any,
        corridor_ctx: Any,
        budget: PriorArmBudget,
        *,
        seed: Optional[int] = None,
        aggregate_counts: bool = True,
        task_id: Optional[str] = None,
    ) -> None:
        if env_ctx is None or corridor_ctx is None:
            # Arm P is only meaningful against the same env+corridor prior
            # that produced the quantum arm's theta. A fallback to the
            # env-blind uniform sampler would silently compare a different
            # prior, so refuse instead.
            raise ValueError(
                "PriorArmBaseSampler requires both env_ctx and corridor_ctx; "
                "the env-blind fallback prior is a different distribution "
                "and would invalidate the Q/P contrast."
            )
        self.env_ctx = env_ctx
        self.corridor_ctx = corridor_ctx
        self.budget = budget
        self.seed = seed
        self.aggregate_counts = bool(aggregate_counts)
        self.task_id = task_id
        self.last_stats: Optional[PriorArmStats] = None

    # ------------------------------------------------------------------ #
    def sample(
        self,
        context_or_inputs,
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[CandidateSample]:
        encoder_inputs = get_encoder_inputs(context_or_inputs)
        eff_seed = seed if seed is not None else self.seed
        if eff_seed is None:
            raise ValueError(
                "PriorArmBaseSampler needs a seed (constructor or per-call); "
                "unseeded arms are not reproducible."
            )

        budget = self.budget
        # A per-call n_samples overrides the configured proposal count so the
        # arm can be driven by QuantumImaginaryTimeSampler's per-tau budget.
        if n_samples is not None and budget.mode is BudgetMode.PROPOSAL:
            budget = PriorArmBudget(
                mode=BudgetMode.PROPOSAL,
                n_proposals=int(n_samples),
                chunk_size=budget.chunk_size,
            )

        # Independent, reproducible chunk seeds from one master seed.
        seed_seq = np.random.SeedSequence(int(eff_seed))

        stats = PriorArmStats(
            budget_mode=budget.mode.value,
            target=(budget.n_proposals if budget.mode is BudgetMode.PROPOSAL
                    else budget.n_valid_target),
        )
        # bitstring -> multiplicity, mirroring the quantum counts representation
        counts: Dict[str, int] = {}

        attempt_cap = self._initial_attempt_cap(budget)

        while True:
            remaining = self._remaining_attempts(budget, stats, attempt_cap)
            if remaining <= 0:
                break
            chunk_n = int(min(budget.chunk_size, remaining))
            child = seed_seq.spawn(1)[0]
            chunk_seed = int(child.generate_state(1, dtype=np.uint32)[0])
            stats.chunk_seeds.append(chunk_seed)

            sampler = PriorConditionedBaseSampler(
                encoder_inputs=encoder_inputs,
                env_ctx=self.env_ctx,
                corridor_ctx=self.corridor_ctx,
                n_prior_samples=chunk_n,
                seed=chunk_seed,
            )
            result = sampler.sample()

            stats.n_chunks += 1
            stats.n_attempts += int(result.sample_count)
            stats.n_valid += int(result.valid_count)
            for reason, k in (result.invalid_reason_counts or {}).items():
                stats.invalid_reason_counts[reason] = (
                    stats.invalid_reason_counts.get(reason, 0) + int(k)
                )
            for bs in result.valid_bitstrings:
                counts[bs] = counts.get(bs, 0) + 1

            # A case-level reachability failure will never produce a path;
            # retrying chunks would spin forever.
            if result.metadata.get("reason") == "case_not_reachable":
                logger.warning(
                    "[arm P] task=%s case not reachable — stopping after "
                    "%d attempts", self.task_id, stats.n_attempts,
                )
                break

            # Refine the VALID-mode cap once a rate has actually been seen.
            if budget.mode is BudgetMode.VALID and stats.n_valid > 0:
                attempt_cap = self._refine_attempt_cap(
                    budget, stats, attempt_cap,
                )

        stats.n_unique = len(counts)
        stats.valid_rate = (
            stats.n_valid / stats.n_attempts if stats.n_attempts else 0.0
        )
        if budget.mode is BudgetMode.VALID:
            target = int(budget.n_valid_target or 0)
            stats.target_met = stats.n_valid >= target
            stats.shortfall = max(0, target - stats.n_valid)
            if not stats.target_met:
                logger.warning(
                    "[arm P] task=%s VALID budget NOT met: %d/%d valid after "
                    "%d attempts (cap %s). Shortfall recorded, not hidden.",
                    self.task_id, stats.n_valid, target,
                    stats.n_attempts, attempt_cap,
                )
        self.last_stats = stats

        return self._to_candidates(counts, stats, eff_seed)

    # ------------------------------------------------------------------ #
    # Budget helpers                                                      #
    # ------------------------------------------------------------------ #
    def _initial_attempt_cap(self, budget: PriorArmBudget) -> int:
        if budget.mode is BudgetMode.PROPOSAL:
            return int(budget.n_proposals or 0)
        if budget.max_attempts:
            return int(budget.max_attempts)
        # No rate observed yet — allow one chunk, then refine.
        return int(budget.chunk_size)

    def _refine_attempt_cap(
        self, budget: PriorArmBudget, stats: PriorArmStats, current_cap: int,
    ) -> int:
        if budget.max_attempts:
            return int(budget.max_attempts)
        rate = stats.n_valid / max(stats.n_attempts, 1)
        if rate <= 0:
            return current_cap
        projected = int(
            np.ceil((budget.n_valid_target or 0) / rate) * ATTEMPT_CAP_SLACK
        )
        return max(current_cap, projected)

    def _remaining_attempts(
        self, budget: PriorArmBudget, stats: PriorArmStats, attempt_cap: int,
    ) -> int:
        if budget.mode is BudgetMode.PROPOSAL:
            return int(budget.n_proposals or 0) - stats.n_attempts
        if stats.n_valid >= int(budget.n_valid_target or 0):
            return 0
        return attempt_cap - stats.n_attempts

    # ------------------------------------------------------------------ #
    def _to_candidates(
        self, counts: Dict[str, int], stats: PriorArmStats, eff_seed: int,
    ) -> List[CandidateSample]:
        """Build the contract-conforming candidate list.

        ``coords`` stays None on purpose — see the module docstring.
        """
        shared = {
            "arm": self.ARM,
            "sampler": "prior_conditioned_rollout",
            "prior_mode": "corridor_bezier",
            "task_id": self.task_id,
            "effective_seed": int(eff_seed),
            "budget_mode": stats.budget_mode,
            "n_attempts": int(stats.n_attempts),
            "n_valid": int(stats.n_valid),
            "n_unique": int(stats.n_unique),
            "valid_rate": float(stats.valid_rate),
            "target_met": bool(stats.target_met),
            "shortfall": int(stats.shortfall),
        }
        out: List[CandidateSample] = []
        if self.aggregate_counts:
            for bs, c in counts.items():
                out.append(CandidateSample(
                    bitstring=bs, codes=None, coords=None,
                    count=int(c), accepted=False, valid=False,
                    metadata=dict(shared),
                ))
        else:
            for bs, c in counts.items():
                for _ in range(int(c)):
                    out.append(CandidateSample(
                        bitstring=bs, codes=None, coords=None,
                        count=1, accepted=False, valid=False,
                        metadata=dict(shared),
                    ))
        return out


__all__ = ["BudgetMode", "PriorArmBudget", "PriorArmStats",
           "PriorArmBaseSampler"]
