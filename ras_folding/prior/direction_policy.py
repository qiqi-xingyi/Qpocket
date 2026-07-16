# Author: Yuqi Zhang
"""PriorConditionedDirectionPolicy — per-step direction probabilities.

Combines:
  - V1 reach + clash + last-bend feasibility mask
  - Environment hard/soft/safe scoring (S2 baseline)
  - Corridor center distance + tangent alignment (C4)

Returns a stable softmax probability vector over 64 lattice directions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


# Defaults validated by V2 segment-corridor simulation
DEFAULT_TEMPERATURE: float = 1.0
DEFAULT_ENV_HARD_THRESHOLD: float = 2.0
DEFAULT_ENV_SOFT_THRESHOLD: float = 3.0
DEFAULT_ENV_HARD_SCORE: float = -10.0
DEFAULT_ENV_SOFT_SCORE: float = -1.0
DEFAULT_ENV_SAFE_SCORE: float = 0.0
DEFAULT_CORRIDOR_WEIGHT: float = 2.0
DEFAULT_CORRIDOR_SIGMA: float = 3.0
DEFAULT_TANGENT_WEIGHT: float = 1.0
CA_CA_LENGTH: float = 3.8


@dataclass
class PriorDirectionResult:
    probabilities: np.ndarray         # (64,) sums to 1.0 (or 0 if invalid)
    total_score: np.ndarray           # (64,)
    env_score: np.ndarray             # (64,)
    corridor_score: np.ndarray        # (64,)
    d_env: np.ndarray                 # (64,)
    feasible_count: int
    entropy: float
    max_probability: float
    uniform_fallback: bool
    invalid: bool                     # True iff feasible_count == 0
    metadata: Dict[str, object] = field(default_factory=dict)


class PriorConditionedDirectionPolicy:
    """Stateless policy. Pass current step + contexts each call."""

    def __init__(
        self,
        env_ctx,
        corridor_ctx,
        *,
        temperature: float = DEFAULT_TEMPERATURE,
        env_hard_threshold: float = DEFAULT_ENV_HARD_THRESHOLD,
        env_soft_threshold: float = DEFAULT_ENV_SOFT_THRESHOLD,
        env_hard_score: float = DEFAULT_ENV_HARD_SCORE,
        env_soft_score: float = DEFAULT_ENV_SOFT_SCORE,
        env_safe_score: float = DEFAULT_ENV_SAFE_SCORE,
        corridor_weight: float = DEFAULT_CORRIDOR_WEIGHT,
        corridor_sigma: float = DEFAULT_CORRIDOR_SIGMA,
        tangent_weight: float = DEFAULT_TANGENT_WEIGHT,
    ):
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0; got {temperature}")
        if corridor_sigma <= 0:
            raise ValueError(f"corridor_sigma must be > 0; got {corridor_sigma}")
        self.env_ctx = env_ctx
        self.corridor_ctx = corridor_ctx
        self.temperature = float(temperature)
        self.env_hard = float(env_hard_threshold)
        self.env_soft = float(env_soft_threshold)
        self.env_hard_score = float(env_hard_score)
        self.env_soft_score = float(env_soft_score)
        self.env_safe_score = float(env_safe_score)
        self.corridor_weight = float(corridor_weight)
        self.corridor_sigma = float(corridor_sigma)
        self.tangent_weight = float(tangent_weight)

    def step_prior(
        self,
        current_pos: np.ndarray,
        last_dir: np.ndarray,
        step_index: int,
        n_bonds: int,
        directions: np.ndarray,        # (64, 3)
        feasible_mask: np.ndarray,     # (64,) bool
    ) -> PriorDirectionResult:
        feas = np.asarray(feasible_mask, dtype=bool)
        n_dirs = int(directions.shape[0])
        endpoints = current_pos[None, :] + CA_CA_LENGTH * directions  # (64,3)
        d_env, _ = self.env_ctx.env_tree.query(endpoints, k=1)
        env_score = self._env_score_from_dist(d_env)
        if step_index + 1 <= n_bonds:
            t_next = (step_index + 1) / float(max(n_bonds, 1))
        else:
            t_next = 1.0
        center = self.corridor_ctx.bezier_point(t_next)
        tangent = self.corridor_ctx.bezier_tangent(t_next)
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm < 1e-12:
            tangent_unit = np.zeros(3)
        else:
            tangent_unit = tangent / tangent_norm
        d_center = np.linalg.norm(endpoints - center[None, :], axis=1)
        s_center = -d_center / self.corridor_sigma
        s_tangent = directions @ tangent_unit
        corridor_score = s_center + self.tangent_weight * s_tangent
        total_score = env_score + self.corridor_weight * corridor_score
        # mask infeasible
        masked = np.where(feas, total_score, -np.inf)
        feasible_count = int(feas.sum())
        invalid = feasible_count == 0
        uniform_fallback = False
        if invalid:
            return PriorDirectionResult(
                probabilities=np.zeros(n_dirs, dtype=np.float64),
                total_score=total_score, env_score=env_score,
                corridor_score=corridor_score, d_env=d_env,
                feasible_count=0, entropy=0.0, max_probability=0.0,
                uniform_fallback=False, invalid=True,
                metadata={"reason": "no_feasible_directions"},
            )
        # stable softmax
        m = float(np.max(masked))
        if not np.isfinite(m):
            # all masked. shouldn't happen — fall back to uniform feasible
            uniform_fallback = True
            probs = np.where(feas, 1.0, 0.0).astype(np.float64)
            probs = probs / probs.sum()
        else:
            shifted = (masked - m) / self.temperature
            ex = np.exp(shifted)
            ex = np.where(feas, ex, 0.0)
            ssum = float(ex.sum())
            if ssum <= 0 or not np.isfinite(ssum):
                uniform_fallback = True
                probs = np.where(feas, 1.0, 0.0).astype(np.float64)
                probs = probs / probs.sum()
            else:
                probs = ex / ssum
        probs = np.where(feas, probs, 0.0)
        ssum = float(probs.sum())
        if ssum <= 0:
            uniform_fallback = True
            probs = np.where(feas, 1.0, 0.0).astype(np.float64)
            probs = probs / probs.sum()
        else:
            probs = probs / ssum
        # entropy
        nz = probs[probs > 0]
        entropy = float(-np.sum(nz * np.log(nz + 1e-30))) if nz.size else 0.0
        return PriorDirectionResult(
            probabilities=probs, total_score=total_score,
            env_score=env_score, corridor_score=corridor_score,
            d_env=d_env, feasible_count=feasible_count,
            entropy=entropy, max_probability=float(probs.max()),
            uniform_fallback=uniform_fallback, invalid=False,
        )

    def _env_score_from_dist(self, d_env: np.ndarray) -> np.ndarray:
        s = np.full_like(d_env, self.env_safe_score, dtype=np.float64)
        hard = d_env < self.env_hard
        soft = (~hard) & (d_env < self.env_soft)
        s[hard] = self.env_hard_score
        s[soft] = self.env_soft_score
        return s


__all__ = [
    "PriorDirectionResult", "PriorConditionedDirectionPolicy",
    "DEFAULT_TEMPERATURE", "DEFAULT_ENV_HARD_THRESHOLD",
    "DEFAULT_ENV_SOFT_THRESHOLD", "DEFAULT_CORRIDOR_WEIGHT",
    "DEFAULT_CORRIDOR_SIGMA", "DEFAULT_TANGENT_WEIGHT",
]
