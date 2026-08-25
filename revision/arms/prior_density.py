# Author: Yuqi Zhang
"""Log-density of a path under the autoregressive prior.

The prior is defined only implicitly, by the rollout in
``ras_folding.prior.prior_sampler``: it draws each bond's direction from a
categorical distribution over the feasible subset of the 64-direction
lattice, conditioned on everything already placed. Evaluating the density
of a given path therefore means replaying that rollout deterministically
under a fixed code sequence and accumulating the per-step log
probabilities.

Every step here mirrors ``PriorConditionedBaseSampler.sample`` exactly --
the same lattice, the same reach/clash/bend masks, the same policy call,
the same normalisation, and the same endpoint tolerance. Any divergence
would mean the Metropolis acceptance ratio targets a distribution other
than the one arm P samples, which is the whole point of the comparison.

A path that leaves the support returns ``-inf`` with the reason recorded,
using the same reason vocabulary as the sampler.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ras_folding.encoder.lattice import (
    BEND_MAX_DEG, BEND_MIN_DEG, N_DIRECTIONS, lattice_around,
)
from ras_folding.encoder.reachable import clash_mask, reach_mask
from ras_folding.encoder.decoder import EPSILON
from ras_folding.prior.direction_policy import PriorConditionedDirectionPolicy
from ras_folding.utils.constants import CA_CA_LENGTH

_COS_BEND_MIN = math.cos(math.radians(BEND_MIN_DEG))
_COS_BEND_MAX = math.cos(math.radians(BEND_MAX_DEG))

NEG_INF = float("-inf")


@dataclass
class PathDensity:
    """Result of evaluating one code sequence."""
    logprob: float
    valid: bool
    reason: Optional[str] = None
    coords: Optional[np.ndarray] = None
    n_steps_evaluated: int = 0
    codes: Optional[List[int]] = None


class PriorPathEvaluator:
    """Replay a code sequence through the prior and score it.

    Construct once per task and reuse: the policy and contexts are fixed,
    only the codes change.
    """

    def __init__(
        self,
        encoder_inputs,
        env_ctx: Any,
        corridor_ctx: Any,
        *,
        policy: Optional[PriorConditionedDirectionPolicy] = None,
        epsilon: float = EPSILON,
    ) -> None:
        self.ei = encoder_inputs
        self.env_ctx = env_ctx
        self.corridor_ctx = corridor_ctx
        self.policy = policy or PriorConditionedDirectionPolicy(
            env_ctx, corridor_ctx,
        )
        self.epsilon = float(epsilon)

        self.n_bonds = int(encoder_inputs.n_bonds)
        self.n_res = int(encoder_inputs.n_residues)
        self.anchor_left = np.asarray(encoder_inputs.anchor_left,
                                      dtype=np.float64)
        self.anchor_right = np.asarray(encoder_inputs.anchor_right,
                                       dtype=np.float64)
        self.v_left_seed = np.asarray(encoder_inputs.v_left_seed,
                                      dtype=np.float64)
        self.v_right_seed = np.asarray(encoder_inputs.v_right_seed,
                                       dtype=np.float64)

    # ------------------------------------------------------------------ #
    def evaluate(self, codes: Sequence[int]) -> PathDensity:
        """Log-density of ``codes`` under the prior."""
        if len(codes) != self.n_bonds:
            return PathDensity(NEG_INF, False, "code_length_mismatch")

        ca = np.zeros((self.n_res, 3), dtype=np.float64)
        ca[0] = self.anchor_left
        last_dir = self.v_left_seed.copy()
        logp = 0.0

        for k in range(self.n_bonds):
            code = int(codes[k])
            if not (0 <= code < N_DIRECTIONS):
                return PathDensity(NEG_INF, False, "code_out_of_range",
                                   n_steps_evaluated=k)

            lat = lattice_around(last_dir)
            p_next = ca[k][None, :] + CA_CA_LENGTH * lat
            remaining = self.n_bonds - k - 1
            rmask = reach_mask(p_next, self.anchor_right, remaining,
                               self.epsilon)
            cmask = clash_mask(p_next, ca[: k + 1])
            if remaining == 0:
                cos_end = lat @ (-self.v_right_seed)
                bmask = ((cos_end >= _COS_BEND_MAX)
                         & (cos_end <= _COS_BEND_MIN))
            else:
                bmask = np.ones(N_DIRECTIONS, dtype=bool)
            feas = rmask & cmask & bmask

            if not feas.any():
                return PathDensity(NEG_INF, False, "feasible_set_empty",
                                   n_steps_evaluated=k)
            if not feas[code]:
                # The move exists in the code space but not in the support.
                return PathDensity(NEG_INF, False, "code_infeasible",
                                   n_steps_evaluated=k)

            pr = self.policy.step_prior(
                current_pos=ca[k], last_dir=last_dir,
                step_index=k, n_bonds=self.n_bonds,
                directions=lat, feasible_mask=feas,
            )
            if pr.invalid:
                return PathDensity(NEG_INF, False, "policy_invalid",
                                   n_steps_evaluated=k)
            p = pr.probabilities
            if not np.isfinite(p).all() or float(p.sum()) <= 0:
                return PathDensity(NEG_INF, False, "policy_nonfinite",
                                   n_steps_evaluated=k)
            p = p / float(p.sum())
            p_code = float(p[code])
            if p_code <= 0.0:
                return PathDensity(NEG_INF, False, "zero_probability_code",
                                   n_steps_evaluated=k)
            logp += math.log(p_code)

            d = lat[code]
            ca[k + 1] = ca[k] + CA_CA_LENGTH * d
            last_dir = d

        # Endpoint tolerance — same check the sampler applies after rollout.
        residual = float(np.linalg.norm(ca[-1] - self.anchor_right))
        if residual > self.epsilon:
            return PathDensity(NEG_INF, False, "endpoint_mismatch",
                               coords=ca, n_steps_evaluated=self.n_bonds)

        return PathDensity(logp, True, None, coords=ca,
                           n_steps_evaluated=self.n_bonds)

    # ------------------------------------------------------------------ #
    def feasible_codes(self, codes: Sequence[int], step: int) -> np.ndarray:
        """Feasible direction codes at ``step`` given the prefix of ``codes``.

        Used by proposals that need to stay inside the support. Returns an
        empty array when the prefix itself is already off-support.
        """
        ca = np.zeros((self.n_res, 3), dtype=np.float64)
        ca[0] = self.anchor_left
        last_dir = self.v_left_seed.copy()
        for k in range(step):
            code = int(codes[k])
            if not (0 <= code < N_DIRECTIONS):
                return np.zeros(0, dtype=np.int64)
            lat = lattice_around(last_dir)
            d = lat[code]
            ca[k + 1] = ca[k] + CA_CA_LENGTH * d
            last_dir = d

        lat = lattice_around(last_dir)
        p_next = ca[step][None, :] + CA_CA_LENGTH * lat
        remaining = self.n_bonds - step - 1
        rmask = reach_mask(p_next, self.anchor_right, remaining, self.epsilon)
        cmask = clash_mask(p_next, ca[: step + 1])
        if remaining == 0:
            cos_end = lat @ (-self.v_right_seed)
            bmask = ((cos_end >= _COS_BEND_MAX) & (cos_end <= _COS_BEND_MIN))
        else:
            bmask = np.ones(N_DIRECTIONS, dtype=bool)
        return np.flatnonzero(rmask & cmask & bmask).astype(np.int64)


    # ------------------------------------------------------------------ #
    def rollout_suffix(
        self, codes: Sequence[int], cut: int, rng: np.random.Generator,
    ) -> PathDensity:
        """Redraw bonds ``cut..n_bonds-1`` from the prior conditional.

        The prefix ``codes[:cut]`` is held fixed and the suffix is sampled
        from exactly the same step-wise conditionals the prior sampler
        uses. This is a Gibbs move on the suffix block: the proposal
        density and the target conditional are the same object, so the
        Metropolis ratio is one for every proposal that lands in the
        support. Proposals that fall out of the support (an empty feasible
        set, or an endpoint outside tolerance) are returned invalid and the
        caller holds the chain in place -- a holding probability, not a
        distortion of the target.

        ``cut=0`` regenerates the whole path and is therefore an
        independent prior draw; the mixture kernel avoids it so the move
        stays local.
        """
        if not (0 <= cut <= self.n_bonds):
            return PathDensity(NEG_INF, False, "cut_out_of_range")

        ca = np.zeros((self.n_res, 3), dtype=np.float64)
        ca[0] = self.anchor_left
        last_dir = self.v_left_seed.copy()
        new_codes: List[int] = []
        logp = 0.0

        for k in range(self.n_bonds):
            lat = lattice_around(last_dir)
            p_next = ca[k][None, :] + CA_CA_LENGTH * lat
            remaining = self.n_bonds - k - 1
            rmask = reach_mask(p_next, self.anchor_right, remaining,
                               self.epsilon)
            cmask = clash_mask(p_next, ca[: k + 1])
            if remaining == 0:
                cos_end = lat @ (-self.v_right_seed)
                bmask = ((cos_end >= _COS_BEND_MAX)
                         & (cos_end <= _COS_BEND_MIN))
            else:
                bmask = np.ones(N_DIRECTIONS, dtype=bool)
            feas = rmask & cmask & bmask
            if not feas.any():
                return PathDensity(NEG_INF, False, "feasible_set_empty",
                                   n_steps_evaluated=k)

            pr = self.policy.step_prior(
                current_pos=ca[k], last_dir=last_dir,
                step_index=k, n_bonds=self.n_bonds,
                directions=lat, feasible_mask=feas,
            )
            if pr.invalid:
                return PathDensity(NEG_INF, False, "policy_invalid",
                                   n_steps_evaluated=k)
            p = pr.probabilities
            if not np.isfinite(p).all() or float(p.sum()) <= 0:
                return PathDensity(NEG_INF, False, "policy_nonfinite",
                                   n_steps_evaluated=k)
            p = p / float(p.sum())

            if k < cut:
                code = int(codes[k])
                if not (0 <= code < N_DIRECTIONS) or not feas[code]:
                    return PathDensity(NEG_INF, False, "prefix_infeasible",
                                       n_steps_evaluated=k)
            else:
                # Same inverse-CDF draw the prior sampler performs.
                cdf = np.cumsum(p)
                code = int(np.searchsorted(cdf, float(rng.random())))
                if code >= N_DIRECTIONS:
                    code = N_DIRECTIONS - 1

            p_code = float(p[code])
            if p_code <= 0.0:
                return PathDensity(NEG_INF, False, "zero_probability_code",
                                   n_steps_evaluated=k)
            logp += math.log(p_code)
            new_codes.append(code)
            d = lat[code]
            ca[k + 1] = ca[k] + CA_CA_LENGTH * d
            last_dir = d

        residual = float(np.linalg.norm(ca[-1] - self.anchor_right))
        if residual > self.epsilon:
            return PathDensity(NEG_INF, False, "endpoint_mismatch",
                               coords=ca, n_steps_evaluated=self.n_bonds)

        out = PathDensity(logp, True, None, coords=ca,
                          n_steps_evaluated=self.n_bonds)
        out.codes = new_codes          # type: ignore[attr-defined]
        return out


__all__ = ["PathDensity", "PriorPathEvaluator", "NEG_INF"]
