# Author: Yuqi Zhang
"""PriorConditionedBaseSampler — path rollout under C4 + B1 prior.

Generates `n_prior_samples` deployable paths through V1 lattice +
reachable mask, scored by env + corridor priors. Returns:
  - the bitstrings of valid paths (6-bit per bond, MSB-first)
  - per-qubit bit marginals consumed by the HEA moment-match initializer
  - per-step diagnostics

Uses ras_folding.encoder.lattice / reachable / decoder constants
(unchanged) so V1 backward compatibility is preserved.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ras_folding.encoder.lattice import (
    BEND_MAX_DEG, BEND_MIN_DEG, N_DIRECTIONS, lattice_around,
)
from ras_folding.encoder.reachable import (
    L_MAX, L_MIN, MIN_SEP, SKIP_RECENT,
    case_reachable, clash_mask, reach_mask,
)
from ras_folding.encoder.decoder import BITS_PER_BOND, EPSILON
from ras_folding.utils.constants import CA_CA_LENGTH

from ras_folding.prior.direction_policy import (
    PriorConditionedDirectionPolicy,
)

import math
_COS_BEND_MIN = math.cos(math.radians(BEND_MIN_DEG))
_COS_BEND_MAX = math.cos(math.radians(BEND_MAX_DEG))


@dataclass
class PriorSamplingResult:
    valid_bitstrings: List[str]
    valid_coords: np.ndarray              # (n_valid, n_residues, 3)
    bit_marginals: np.ndarray             # (n_qubits,) ∈ [0, 1]
    n_qubits: int
    sample_count: int
    valid_count: int
    invalid_count: int
    invalid_reason_counts: Dict[str, int]
    step_stats: List[Dict[str, float]]    # per-bond aggregated stats
    path_stats: Dict[str, float]
    seed: Optional[int]
    fallback_to_uniform: bool             # True if valid_count too low
    metadata: Dict[str, object] = field(default_factory=dict)


def _last_bend_ok(direction: np.ndarray, v_right_seed: np.ndarray) -> bool:
    target = -v_right_seed
    cos_end = float(np.dot(direction, target))
    return (cos_end >= _COS_BEND_MAX) and (cos_end <= _COS_BEND_MIN)


class PriorConditionedBaseSampler:
    def __init__(
        self,
        encoder_inputs,
        env_ctx,
        corridor_ctx,
        *,
        n_prior_samples: int = 256,
        seed: Optional[int] = None,
        policy: Optional[PriorConditionedDirectionPolicy] = None,
        epsilon: float = EPSILON,
        bits_per_step: int = BITS_PER_BOND,
        fallback_min_valid: int = 8,
    ):
        self.encoder_inputs = encoder_inputs
        self.env_ctx = env_ctx
        self.corridor_ctx = corridor_ctx
        self.n_prior_samples = int(n_prior_samples)
        self.seed = seed
        self.policy = policy or PriorConditionedDirectionPolicy(
            env_ctx, corridor_ctx,
        )
        self.epsilon = float(epsilon)
        self.bits_per_step = int(bits_per_step)
        self.fallback_min_valid = int(fallback_min_valid)

    def sample(self) -> PriorSamplingResult:
        rng = np.random.default_rng(self.seed)
        ei = self.encoder_inputs
        n_bonds = int(ei.n_bonds)
        n_res = int(ei.n_residues)
        n_qubits = n_bonds * self.bits_per_step
        anchor_left = np.asarray(ei.anchor_left, dtype=np.float64)
        anchor_right = np.asarray(ei.anchor_right, dtype=np.float64)
        v_left_seed = np.asarray(ei.v_left_seed, dtype=np.float64)
        v_right_seed = np.asarray(ei.v_right_seed, dtype=np.float64)

        # case-level reach precheck
        if not case_reachable(ei, self.epsilon):
            return PriorSamplingResult(
                valid_bitstrings=[],
                valid_coords=np.zeros((0, n_res, 3), dtype=np.float64),
                bit_marginals=np.full(n_qubits, 0.5, dtype=np.float64),
                n_qubits=n_qubits, sample_count=0, valid_count=0,
                invalid_count=0,
                invalid_reason_counts={"case_not_reachable": 1},
                step_stats=[], path_stats={},
                seed=self.seed,
                fallback_to_uniform=True,
                metadata={"reason": "case_not_reachable"},
            )

        valid_bitstrings: List[str] = []
        valid_coords: List[np.ndarray] = []
        invalid_reason_counts: Dict[str, int] = {}
        # Per-step accumulators
        step_feasible_count: List[List[int]] = [[] for _ in range(n_bonds)]
        step_entropy: List[List[float]] = [[] for _ in range(n_bonds)]
        step_max_prob: List[List[float]] = [[] for _ in range(n_bonds)]
        step_d_env: List[List[float]] = [[] for _ in range(n_bonds)]
        step_corridor: List[List[float]] = [[] for _ in range(n_bonds)]

        for sample_idx in range(self.n_prior_samples):
            ca = np.zeros((n_res, 3), dtype=np.float64)
            ca[0] = anchor_left
            last_dir = v_left_seed.copy()
            bit_chunks: List[str] = []
            valid = True
            invalid_reason = None
            for k in range(n_bonds):
                lat = lattice_around(last_dir)              # (64, 3)
                p_next = ca[k][None, :] + CA_CA_LENGTH * lat
                remaining = n_bonds - k - 1
                rmask = reach_mask(p_next, anchor_right, remaining,
                                    self.epsilon)
                cmask = clash_mask(p_next, ca[: k + 1])
                if remaining == 0:
                    target = -v_right_seed
                    cos_end = lat @ target
                    bmask = ((cos_end >= _COS_BEND_MAX)
                             & (cos_end <= _COS_BEND_MIN))
                else:
                    bmask = np.ones(N_DIRECTIONS, dtype=bool)
                feas = rmask & cmask & bmask
                if not feas.any():
                    valid = False
                    invalid_reason = "feasible_set_empty"
                    break
                pr = self.policy.step_prior(
                    current_pos=ca[k], last_dir=last_dir,
                    step_index=k, n_bonds=n_bonds,
                    directions=lat, feasible_mask=feas,
                )
                if pr.invalid:
                    valid = False
                    invalid_reason = "policy_invalid"
                    break
                # categorical sample
                p = pr.probabilities
                if not np.isfinite(p).all() or float(p.sum()) <= 0:
                    valid = False
                    invalid_reason = "policy_nonfinite"
                    break
                # ensure normalised
                p = p / float(p.sum())
                u = float(rng.random())
                cdf = np.cumsum(p)
                idx = int(np.searchsorted(cdf, u))
                if idx >= N_DIRECTIONS:
                    idx = N_DIRECTIONS - 1
                # update bit chunk for this bond — 6-bit MSB-first
                code = idx & ((1 << self.bits_per_step) - 1)
                bit_chunks.append(format(code, f"0{self.bits_per_step}b"))
                # record step stats
                step_feasible_count[k].append(pr.feasible_count)
                step_entropy[k].append(pr.entropy)
                step_max_prob[k].append(pr.max_probability)
                step_d_env[k].append(float(pr.d_env[idx]))
                step_corridor[k].append(float(pr.corridor_score[idx]))
                # advance
                d = lat[idx]
                ca[k + 1] = ca[k] + CA_CA_LENGTH * d
                last_dir = d
            if valid:
                # endpoint tolerance check (mirrors decode_bitstring_with_info)
                ep_residual = float(np.linalg.norm(
                    ca[-1] - anchor_right
                ))
                if ep_residual > self.epsilon:
                    valid = False
                    invalid_reason = "endpoint_mismatch"
            if valid:
                bitstring = "".join(bit_chunks)
                valid_bitstrings.append(bitstring)
                valid_coords.append(ca)
            else:
                invalid_reason_counts[invalid_reason] = (
                    invalid_reason_counts.get(invalid_reason, 0) + 1
                )

        valid_count = len(valid_bitstrings)
        invalid_count = self.n_prior_samples - valid_count

        # bit marginals over valid paths
        if valid_count > 0:
            bm = np.zeros(n_qubits, dtype=np.float64)
            for bs in valid_bitstrings:
                if len(bs) != n_qubits:
                    continue
                arr = np.frombuffer(
                    bs.encode("ascii"), dtype=np.uint8
                ) - ord("0")
                bm += arr.astype(np.float64)
            bm = bm / float(valid_count)
            fallback_to_uniform = valid_count < self.fallback_min_valid
            if fallback_to_uniform:
                # blend with 0.5 baseline
                bm = 0.5 * bm + 0.5 * 0.5
        else:
            bm = np.full(n_qubits, 0.5, dtype=np.float64)
            fallback_to_uniform = True

        coords_arr = (np.stack(valid_coords, axis=0)
                      if valid_count > 0
                      else np.zeros((0, n_res, 3), dtype=np.float64))

        step_stats = []
        for k in range(n_bonds):
            def _mean_or_nan(xs):
                return float(np.mean(xs)) if xs else float("nan")
            step_stats.append({
                "step_index": k,
                "feasible_count_mean": _mean_or_nan(step_feasible_count[k]),
                "entropy_mean": _mean_or_nan(step_entropy[k]),
                "max_prob_mean": _mean_or_nan(step_max_prob[k]),
                "d_env_mean": _mean_or_nan(step_d_env[k]),
                "d_env_min": (float(np.min(step_d_env[k]))
                               if step_d_env[k] else float("nan")),
                "corridor_score_mean": _mean_or_nan(step_corridor[k]),
                "n_paths_passing": len(step_feasible_count[k]),
            })
        path_stats = {
            "valid_rate":
                valid_count / max(self.n_prior_samples, 1),
            "fallback_to_uniform": bool(fallback_to_uniform),
            "n_qubits": int(n_qubits),
            "n_bonds": int(n_bonds),
        }
        return PriorSamplingResult(
            valid_bitstrings=valid_bitstrings,
            valid_coords=coords_arr,
            bit_marginals=bm,
            n_qubits=n_qubits,
            sample_count=self.n_prior_samples,
            valid_count=valid_count,
            invalid_count=invalid_count,
            invalid_reason_counts=invalid_reason_counts,
            step_stats=step_stats,
            path_stats=path_stats,
            seed=self.seed,
            fallback_to_uniform=fallback_to_uniform,
        )


__all__ = [
    "PriorSamplingResult",
    "PriorConditionedBaseSampler",
]
