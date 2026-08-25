# Author: Yuqi Zhang
"""MCMCArmBaseSampler (arm M) — Metropolis-Hastings targeting the prior.

Arm M is the classical-sampler control in the same discrete space. Its
target is the prior itself, so any difference between arm M and arm P is a
property of the sampling mechanism -- how a local Markov chain explores
the space versus independent draws -- and not of a different objective.

Why the chain is not independence MH
------------------------------------
Independence MH with the prior as the proposal and the prior as the target
accepts with probability one at every step and returns i.i.d. prior draws.
That is arm P exactly, so it is not a distinct arm. A meaningful M must
move locally, which is what the two kernels below do.

Kernel
------
A mixture of two moves, both leaving the prior invariant:

``single_site``
    Pick one bond uniformly and redraw its direction code uniformly from
    the 64-direction lattice. The proposal is symmetric, so the acceptance
    ratio is the prior density ratio. Changing bond *j* displaces every
    downstream position, so most such proposals leave the support and are
    rejected; the resulting acceptance rate is a measured property of the
    space, not a defect to be tuned away.

``suffix``
    Pick a cut point and redraw the whole suffix from the prior
    conditional, holding the prefix fixed. Proposal and target conditional
    are the same object, so every proposal landing in the support is
    accepted. The cut point is drawn from ``1..n_bonds-1``: a cut at zero
    would regenerate the entire path and collapse the move to an
    independent prior draw, which is arm P.

Proposals that leave the support are rejected and the chain holds. The
holding probability is part of the kernel and does not bias the target.

Accounting
----------
A proposal is one attempt, matching one quantum shot. Accepted moves,
rejections by cause, and the number of distinct states visited are all
recorded: a chain that proposes two million times and visits few distinct
states has not covered what a two-million-shot budget suggests, and the
run record must show that rather than reporting the proposal count alone.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ras_folding.encoder.lattice import N_DIRECTIONS
from ras_folding.sampler.context import get_encoder_inputs
from ras_folding.sampler.sample_types import (
    CandidateSample, codes_to_bitstring_str,
)
from revision.arms.prior_arm import BudgetMode, PriorArmBudget
from revision.arms.prior_density import PriorPathEvaluator

logger = logging.getLogger(__name__)

DEFAULT_SUFFIX_MOVE_PROB = 0.5
DEFAULT_BURN_IN = 1000


class MoveType(str, enum.Enum):
    SINGLE_SITE = "single_site"
    SUFFIX = "suffix"


@dataclass
class MCMCArmStats:
    n_proposals: int = 0
    n_accepted: int = 0
    n_unique: int = 0
    n_burn_in: int = 0
    n_recorded: int = 0
    acceptance_rate: float = 0.0
    by_move: Dict[str, Dict[str, int]] = field(default_factory=dict)
    reject_reasons: Dict[str, int] = field(default_factory=dict)
    init_attempts: int = 0
    chain_initialized: bool = True
    budget_mode: str = ""
    target: Optional[int] = None
    target_met: bool = True
    shortfall: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_proposals": int(self.n_proposals),
            "n_accepted": int(self.n_accepted),
            "n_unique": int(self.n_unique),
            "n_burn_in": int(self.n_burn_in),
            "n_recorded": int(self.n_recorded),
            "acceptance_rate": float(self.acceptance_rate),
            "by_move": {k: dict(v) for k, v in self.by_move.items()},
            "reject_reasons": dict(self.reject_reasons),
            "init_attempts": int(self.init_attempts),
            "chain_initialized": bool(self.chain_initialized),
            "budget_mode": self.budget_mode,
            "target": self.target,
            "target_met": bool(self.target_met),
            "shortfall": int(self.shortfall),
        }


class MCMCArmBaseSampler:
    """Arm M. Same contract as ``QuantumBackendBaseSampler.sample``."""

    ARM = "M"

    def __init__(
        self,
        env_ctx: Any,
        corridor_ctx: Any,
        budget: PriorArmBudget,
        *,
        seed: Optional[int] = None,
        suffix_move_prob: float = DEFAULT_SUFFIX_MOVE_PROB,
        burn_in: int = DEFAULT_BURN_IN,
        thin: int = 1,
        max_init_attempts: int = 100_000,
        task_id: Optional[str] = None,
    ) -> None:
        if env_ctx is None or corridor_ctx is None:
            raise ValueError(
                "MCMCArmBaseSampler requires both env_ctx and corridor_ctx: "
                "arm M must target the same prior as arms P and Q."
            )
        if not 0.0 <= suffix_move_prob <= 1.0:
            raise ValueError(
                f"suffix_move_prob must be in [0, 1]; got {suffix_move_prob}")
        if thin < 1:
            raise ValueError(f"thin must be >= 1; got {thin}")
        self.env_ctx = env_ctx
        self.corridor_ctx = corridor_ctx
        self.budget = budget
        self.seed = seed
        self.suffix_move_prob = float(suffix_move_prob)
        self.burn_in = int(burn_in)
        self.thin = int(thin)
        self.max_init_attempts = int(max_init_attempts)
        self.task_id = task_id
        self.last_stats: Optional[MCMCArmStats] = None

    # ------------------------------------------------------------------ #
    def sample(
        self,
        context_or_inputs,
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[CandidateSample]:
        ei = get_encoder_inputs(context_or_inputs)
        eff_seed = seed if seed is not None else self.seed
        if eff_seed is None:
            raise ValueError(
                "MCMCArmBaseSampler needs a seed (constructor or per-call); "
                "unseeded chains are not reproducible."
            )
        budget = self.budget
        if n_samples is not None and budget.mode is BudgetMode.PROPOSAL:
            budget = PriorArmBudget(
                mode=BudgetMode.PROPOSAL, n_proposals=int(n_samples),
                chunk_size=budget.chunk_size,
            )

        rng = np.random.default_rng(int(eff_seed))
        ev = PriorPathEvaluator(ei, self.env_ctx, self.corridor_ctx)
        n_bonds = int(ei.n_bonds)

        stats = MCMCArmStats(
            budget_mode=budget.mode.value,
            n_burn_in=self.burn_in,
            target=(budget.n_proposals if budget.mode is BudgetMode.PROPOSAL
                    else budget.n_valid_target),
            by_move={m.value: {"proposed": 0, "accepted": 0}
                     for m in MoveType},
        )

        # --- initialise the chain inside the support -------------------
        cur = ev.rollout_suffix([], 0, rng)
        stats.init_attempts = 1
        while not cur.valid and stats.init_attempts < self.max_init_attempts:
            cur = ev.rollout_suffix([], 0, rng)
            stats.init_attempts += 1
        if not cur.valid:
            stats.chain_initialized = False
            stats.target_met = False
            stats.shortfall = int(stats.target or 0)
            logger.error(
                "[arm M] task=%s could not initialise inside the support "
                "after %d attempts — no chain was run.",
                self.task_id, stats.init_attempts,
            )
            self.last_stats = stats
            return []

        cur_codes: List[int] = list(cur.codes or [])
        cur_logp = float(cur.logprob)
        counts: Dict[str, int] = {}
        step = 0

        while True:
            if not self._should_continue(budget, stats, counts):
                break
            step += 1
            stats.n_proposals += 1

            use_suffix = (n_bonds > 1
                          and float(rng.random()) < self.suffix_move_prob)
            if use_suffix:
                move = MoveType.SUFFIX
                cut = int(rng.integers(1, n_bonds))
                prop = ev.rollout_suffix(cur_codes, cut, rng)
                prop_codes = list(prop.codes or [])
                # Gibbs on the suffix block: ratio is one inside the support.
                accept = prop.valid
            else:
                move = MoveType.SINGLE_SITE
                j = int(rng.integers(0, n_bonds))
                prop_codes = list(cur_codes)
                prop_codes[j] = int(rng.integers(0, N_DIRECTIONS))
                prop = ev.evaluate(prop_codes)
                if not prop.valid:
                    accept = False
                else:
                    # Symmetric proposal — Metropolis on the density ratio.
                    log_alpha = float(prop.logprob) - cur_logp
                    accept = (log_alpha >= 0.0
                              or float(rng.random()) < np.exp(log_alpha))

            stats.by_move[move.value]["proposed"] += 1
            if accept:
                stats.by_move[move.value]["accepted"] += 1
                stats.n_accepted += 1
                cur_codes = prop_codes
                cur_logp = float(prop.logprob)
            elif not prop.valid:
                reason = prop.reason or "unknown"
                stats.reject_reasons[reason] = (
                    stats.reject_reasons.get(reason, 0) + 1)
            else:
                stats.reject_reasons["metropolis_reject"] = (
                    stats.reject_reasons.get("metropolis_reject", 0) + 1)

            # Record the chain state, burn-in and thinning applied. The
            # current state is recorded whether or not the last move was
            # accepted: holding is part of the chain, and dropping held
            # states would distort the sampled distribution.
            if step > self.burn_in and (step - self.burn_in) % self.thin == 0:
                bs = codes_to_bitstring_str(cur_codes, n_bonds)
                counts[bs] = counts.get(bs, 0) + 1
                stats.n_recorded += 1

        stats.n_unique = len(counts)
        stats.acceptance_rate = (
            stats.n_accepted / stats.n_proposals if stats.n_proposals else 0.0)
        if budget.mode is BudgetMode.VALID:
            target = int(budget.n_valid_target or 0)
            stats.target_met = stats.n_unique >= target
            stats.shortfall = max(0, target - stats.n_unique)
            if not stats.target_met:
                logger.warning(
                    "[arm M] task=%s VALID budget NOT met: %d/%d distinct "
                    "states after %d proposals. Shortfall recorded.",
                    self.task_id, stats.n_unique, target, stats.n_proposals,
                )
        self.last_stats = stats

        shared = {
            "arm": self.ARM,
            "sampler": "metropolis_hastings_prior_target",
            "task_id": self.task_id,
            "effective_seed": int(eff_seed),
            "suffix_move_prob": self.suffix_move_prob,
            "burn_in": self.burn_in,
            "thin": self.thin,
            "budget_mode": stats.budget_mode,
            "n_proposals": int(stats.n_proposals),
            "n_accepted": int(stats.n_accepted),
            "acceptance_rate": float(stats.acceptance_rate),
            "n_unique": int(stats.n_unique),
        }
        return [
            CandidateSample(bitstring=bs, codes=None, coords=None,
                            count=int(c), accepted=False, valid=False,
                            metadata=dict(shared))
            for bs, c in counts.items()
        ]

    # ------------------------------------------------------------------ #
    def _should_continue(
        self, budget: PriorArmBudget, stats: MCMCArmStats,
        counts: Dict[str, int],
    ) -> bool:
        if budget.mode is BudgetMode.PROPOSAL:
            return stats.n_proposals < int(budget.n_proposals or 0)
        if len(counts) >= int(budget.n_valid_target or 0):
            return False
        cap = budget.max_attempts
        if cap and stats.n_proposals >= int(cap):
            return False
        return True


__all__ = ["MoveType", "MCMCArmStats", "MCMCArmBaseSampler"]
